# filename: codebase/data_processing.py
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix, get_laplacian, to_dense_adj
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import scipy.sparse
from scipy.sparse.linalg import eigs
from scipy.stats import skew, kurtosis
import os
import time

# --- Configuration ---
DATABASE_PATH = "data/"
F_TREE = '/Users/fvillaescusa/Documents/Software/AstroPilot/data/Pablo_merger_trees2.pt'
NUM_SMALLEST_EIG_SUM = 10  # Number of smallest non-zero Laplacian eigenvalues to sum
NUM_DIFFUSION_EIGENVECTORS = 5  # Number of eigenvectors for diffusion map

# --- Helper Functions ---

def extract_features_for_graph(graph_data, scaler_x_fitted):
    """
    Extracts engineered features for a single graph.

    Args:
        graph_data (torch_geometric.data.Data): The input graph data.
        scaler_x_fitted (sklearn.preprocessing.StandardScaler): Fitted scaler for node features.

    Returns:
        np.ndarray: A 1D NumPy array containing the concatenated engineered features.
                    Returns an array of NaNs if critical errors occur.
    """
    num_nodes = graph_data.num_nodes
    edge_index = graph_data.edge_index
    
    # Normalize node features for edge feature calculation
    # Note: Original graph_data.x is used if scaler_x_fitted is None or not applied here.
    # For this step, we assume graph_data.x should be normalized for consistency if edge features depend on scaled values.
    # The main script will pass graphs with already normalized .x if that's the final decision.
    # For now, let's normalize a copy for calculations if scaler is provided.
    x_original = graph_data.x.numpy()
    try:
        x_norm = scaler_x_fitted.transform(x_original)
    except Exception as e:
        # This might happen if x_original has an unexpected shape (e.g. 0 nodes)
        # Fallback to using original x or handle as NaN features. For now, let's signal error.
        # The number of features needs to be consistent.
        # Let's define the total number of features expected.
        # Spectral (4 moments + 1 sum) + Diffusion (NUM_DIFFUSION_EIGENVECTORS * 3) + Edge (4)
        # = 5 + NUM_DIFFUSION_EIGENVECTORS * 3 + 4
        num_total_features = 5 + NUM_DIFFUSION_EIGENVECTORS * 3 + 4
        return np.full(num_total_features, np.nan)


    # 1. Edge-Level Features (using normalized node features)
    # x_norm columns: 0:log10(mass), 1:log10(concentration), 2:log10(Vmax), 3:scale_factor
    delta_sf_list = []
    log_mass_ratio_list = []
    
    if edge_index.shape[1] > 0:
        sources, targets = edge_index[0].numpy(), edge_index[1].numpy()
        # Assuming edges point from progenitor (source) to descendant (target)
        # sf_descendant - sf_progenitor
        # log_mass_descendant - log_mass_progenitor
        # These are normalized values
        sf_diffs = x_norm[targets, 3] - x_norm[sources, 3] 
        mass_ratio_logs = x_norm[targets, 0] - x_norm[sources, 0]

        delta_sf_list = np.abs(sf_diffs) # Already sf_v - sf_u, should be positive. abs for safety.
        log_mass_ratio_list = mass_ratio_logs # Should be positive.

        mean_delta_sf = np.mean(delta_sf_list) if len(delta_sf_list) > 0 else np.nan
        var_delta_sf = np.var(delta_sf_list) if len(delta_sf_list) > 0 else np.nan
        mean_log_mass_ratio = np.mean(log_mass_ratio_list) if len(log_mass_ratio_list) > 0 else np.nan
        var_log_mass_ratio = np.var(log_mass_ratio_list) if len(log_mass_ratio_list) > 0 else np.nan
    else:
        mean_delta_sf, var_delta_sf, mean_log_mass_ratio, var_log_mass_ratio = [np.nan] * 4
    
    edge_features = np.array([mean_delta_sf, var_delta_sf, mean_log_mass_ratio, var_log_mass_ratio])

    # 2. Graph Laplacian Spectral Features
    laplacian_eigenvalues_all = []
    if num_nodes > 0:
        try:
            l_edge_index, l_edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)
            # Convert to dense matrix for eigenvalue computation
            # Ensure l_edge_weight is float
            if l_edge_weight is not None:
                l_edge_weight = l_edge_weight.float()
            
            # Handle case of graph with 0 edges but >0 nodes (get_laplacian returns L_ii=0)
            if l_edge_index.shape[1] == 0 and num_nodes > 0:
                 laplacian_eigenvalues_all = np.zeros(num_nodes)
            else:
                lap_matrix_dense = to_dense_adj(l_edge_index, edge_attr=l_edge_weight, max_num_nodes=num_nodes)[0].numpy()
                laplacian_eigenvalues_all = np.linalg.eigvalsh(lap_matrix_dense)

        except Exception as e:
            laplacian_eigenvalues_all = np.full(num_nodes if num_nodes > 0 else 1, np.nan)
    
    if not isinstance(laplacian_eigenvalues_all, np.ndarray) or laplacian_eigenvalues_all.size == 0:
        spectral_moments = np.full(4, np.nan)
        sum_smallest_nonzero_eig = np.nan
    elif num_nodes == 1:
        spectral_moments = np.array([0.0, 0.0, 0.0, 0.0])
        sum_smallest_nonzero_eig = 0.0
    else:
        mean_eig = np.mean(laplacian_eigenvalues_all)
        std_eig = np.std(laplacian_eigenvalues_all)
        skew_eig = skew(laplacian_eigenvalues_all) if std_eig > 1e-9 else 0.0
        kurt_eig = kurtosis(laplacian_eigenvalues_all, fisher=True) if std_eig > 1e-9 else 0.0
        spectral_moments = np.array([mean_eig, std_eig, skew_eig, kurt_eig])
        
        non_zero_eigs = laplacian_eigenvalues_all[laplacian_eigenvalues_all > 1e-9]
        non_zero_eigs_sorted = np.sort(non_zero_eigs)
        sum_smallest_nonzero_eig = np.sum(non_zero_eigs_sorted[:NUM_SMALLEST_EIG_SUM])

    laplacian_features = np.concatenate([spectral_moments, np.array([sum_smallest_nonzero_eig])])

    # 3. Diffusion Map Features
    diffusion_aggregated_features = np.full(NUM_DIFFUSION_EIGENVECTORS * 3, np.nan)
    if num_nodes >= 2:
        try:
            adj_matrix = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).astype(float)
            out_degree = np.array(adj_matrix.sum(axis=1)).flatten()
            
            inv_out_degree = np.zeros_like(out_degree, dtype=float)
            non_zero_mask = out_degree > 0
            inv_out_degree[non_zero_mask] = 1.0 / out_degree[non_zero_mask]
            
            D_out_inv = scipy.sparse.diags(inv_out_degree)
            P = D_out_inv @ adj_matrix
            
            k_diff_eig = min(NUM_DIFFUSION_EIGENVECTORS, num_nodes - 1)

            if k_diff_eig > 0:
                eigvals_P, eigvecs_P = eigs(P, k=k_diff_eig, which='LR')
                sorted_indices = np.argsort(-np.real(eigvals_P))
                eigvecs_P_sorted_real = np.real(eigvecs_P[:, sorted_indices])

                node_embeddings_diffusion = np.full((num_nodes, NUM_DIFFUSION_EIGENVECTORS), np.nan)
                node_embeddings_diffusion[:, :eigvecs_P_sorted_real.shape[1]] = eigvecs_P_sorted_real

                mean_pooled_diff = np.mean(node_embeddings_diffusion, axis=0)
                max_pooled_diff = np.max(node_embeddings_diffusion, axis=0)
                min_pooled_diff = np.min(node_embeddings_diffusion, axis=0)
                diffusion_aggregated_features = np.concatenate([mean_pooled_diff, max_pooled_diff, min_pooled_diff])
        except Exception as e:
            pass

    all_features = np.concatenate([edge_features, laplacian_features, diffusion_aggregated_features])
    return all_features

# --- Main Script ---
if __name__ == '__main__':
    os.makedirs(DATABASE_PATH, exist_ok=True)

    print("Loading dataset...")
    try:
        full_dataset = torch.load(F_TREE) 
    except Exception as e:
        print("Error loading dataset:" + str(e))
        print("Attempting to load with map_location='cpu'...")
        full_dataset = torch.load(F_TREE, map_location=torch.device('cpu'))

    print("Dataset loaded. Number of graphs: " + str(len(full_dataset)))

    all_lh_ids = np.array([data.lh_id.item() for data in full_dataset])
    unique_lh_ids = np.unique(all_lh_ids)
    print("Number of unique simulation IDs (lh_id): " + str(len(unique_lh_ids)))

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_sim_idx, test_sim_idx = next(splitter.split(unique_lh_ids, groups=unique_lh_ids))
    
    train_simulation_ids = unique_lh_ids[train_sim_idx]
    test_simulation_ids = unique_lh_ids[test_sim_idx]

    train_indices = [i for i, lh_id in enumerate(all_lh_ids) if lh_id in train_simulation_ids]
    test_indices = [i for i, lh_id in enumerate(all_lh_ids) if lh_id in test_simulation_ids]

    print("Number of training simulations: " + str(len(train_simulation_ids)))
    print("Number of testing simulations: " + str(len(test_simulation_ids)))
    print("Number of training graphs: " + str(len(train_indices)))
    print("Number of testing graphs: " + str(len(test_indices)))

    print("Normalizing node features...")
    all_train_x_list = []
    for i in train_indices:
        if hasattr(full_dataset[i], 'x') and full_dataset[i].x is not None and full_dataset[i].x.shape[0] > 0:
            all_train_x_list.append(full_dataset[i].x)
        else:
            print("Warning: Graph at index " + str(i) + " in training set has no nodes or no 'x' attribute. Skipping for scaler fitting.")

    if not all_train_x_list:
        print("Error: No valid node features found in the training set to fit the scaler.")
        scaler_x = StandardScaler()
        normalization_params = {'mean': np.zeros(4), 'std': np.ones(4)}
    else:
        all_train_x = torch.cat(all_train_x_list, dim=0).numpy()
        scaler_x = StandardScaler()
        scaler_x.fit(all_train_x)
        normalization_params = {'mean': scaler_x.mean_, 'std': scaler_x.scale_}

    norm_params_path = os.path.join(DATABASE_PATH, 'normalization_params.npz')
    np.savez(norm_params_path, **normalization_params)
    print("Normalization parameters saved to " + norm_params_path)
    print("  Mean:" + str(normalization_params['mean']))
    print("  Std:" + str(normalization_params['std']))

    print("Starting feature engineering...")
    
    processed_train_data = {'X': [], 'y': [], 'lh_ids': []}
    processed_test_data = {'X': [], 'y': [], 'lh_ids': []}
    
    start_time_train = time.time()
    num_nan_feature_vectors_train = 0
    for i, graph_idx in enumerate(train_indices):
        original_graph = full_dataset[graph_idx]
        features = extract_features_for_graph(original_graph, scaler_x)
        
        if np.isnan(features).all():
            num_nan_feature_vectors_train +=1
            print("Warning: All features are NaN for training graph index " + str(graph_idx) + " (lh_id: " + str(original_graph.lh_id.item()) + "). Skipping.")
            continue

        processed_train_data['X'].append(features)
        processed_train_data['y'].append(original_graph.y.numpy().flatten())
        processed_train_data['lh_ids'].append(original_graph.lh_id.item())
        if (i + 1) % 100 == 0:
            print("  Processed " + str(i+1) + "/" + str(len(train_indices)) + " training graphs...")
    end_time_train = time.time()
    print("Training set feature engineering completed in " + str(end_time_train - start_time_train) + " seconds.")
    if num_nan_feature_vectors_train > 0:
        print("Warning: Skipped " + str(num_nan_feature_vectors_train) + " training graphs due to all NaN features.")


    start_time_test = time.time()
    num_nan_feature_vectors_test = 0
    for i, graph_idx in enumerate(test_indices):
        original_graph = full_dataset[graph_idx]
        features = extract_features_for_graph(original_graph, scaler_x)

        if np.isnan(features).all():
            num_nan_feature_vectors_test +=1
            print("Warning: All features are NaN for test graph index " + str(graph_idx) + " (lh_id: " + str(original_graph.lh_id.item()) + "). Skipping.")
            continue

        processed_test_data['X'].append(features)
        processed_test_data['y'].append(original_graph.y.numpy().flatten())
        processed_test_data['lh_ids'].append(original_graph.lh_id.item())
        if (i + 1) % 50 == 0:
            print("  Processed " + str(i+1) + "/" + str(len(test_indices)) + " testing graphs...")
    end_time_test = time.time()
    print("Testing set feature engineering completed in " + str(end_time_test - start_time_test) + " seconds.")
    if num_nan_feature_vectors_test > 0:
        print("Warning: Skipped " + str(num_nan_feature_vectors_test) + " test graphs due to all NaN features.")

    processed_train_data['X'] = np.array(processed_train_data['X'])
    processed_train_data['y'] = np.array(processed_train_data['y'])
    processed_train_data['lh_ids'] = np.array(processed_train_data['lh_ids'])
    
    processed_test_data['X'] = np.array(processed_test_data['X'])
    processed_test_data['y'] = np.array(processed_test_data['y'])
    processed_test_data['lh_ids'] = np.array(processed_test_data['lh_ids'])

    train_features_path = os.path.join(DATABASE_PATH, 'engineered_features_train.npz')
    np.savez(train_features_path, 
             X=processed_train_data['X'], 
             y=processed_train_data['y'], 
             lh_ids=processed_train_data['lh_ids'])
    print("Engineered training features saved to " + train_features_path)

    test_features_path = os.path.join(DATABASE_PATH, 'engineered_features_test.npz')
    np.savez(test_features_path, 
             X=processed_test_data['X'], 
             y=processed_test_data['y'], 
             lh_ids=processed_test_data['lh_ids'])
    print("Engineered testing features saved to " + test_features_path)

    if len(processed_train_data['X']) > 0:
        print("Shape of engineered training features (X_train): " + str(processed_train_data['X'].shape))
        print("Shape of training labels (y_train): " + str(processed_train_data['y'].shape))
        num_nans_train = np.sum(np.isnan(processed_train_data['X']))
        if num_nans_train > 0:
            print("Number of NaN values in training features: " + str(num_nans_train))
            nan_cols_train = np.where(np.isnan(processed_train_data['X']).any(axis=0))[0]
            print("Columns with NaNs in training features: " + str(nan_cols_train))

    if len(processed_test_data['X']) > 0:
        print("Shape of engineered testing features (X_test): " + str(processed_test_data['X'].shape))
        print("Shape of testing labels (y_test): " + str(processed_test_data['y'].shape))
        num_nans_test = np.sum(np.isnan(processed_test_data['X']))
        if num_nans_test > 0:
            print("Number of NaN values in testing features: " + str(num_nans_test))
            nan_cols_test = np.where(np.isnan(processed_test_data['X']).any(axis=0))[0]
            print("Columns with NaNs in testing features: " + str(nan_cols_test))
            
    print("Step 1: Data Loading, Preprocessing, and Feature Engineering completed.")
