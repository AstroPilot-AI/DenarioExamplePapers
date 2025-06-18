# filename: codebase/tda_correlation_analysis.py
import torch
import numpy as np
import pandas as pd
import gudhi as gd
import gudhi.persistence_graphical_tools
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Ensure LaTeX rendering is off
matplotlib.rcParams['text.usetex'] = False


def extract_topological_features(data, sf_min_norm, sf_max_norm):
    r"""
    Extracts topological features from a single merger tree.

    Args:
        data (torch_geometric.data.Data): A single merger tree data object.
            It must contain:
            - x: Node features, where x[:, 3] is the *normalized* scale factor.
            - edge_index: Edge connectivity.
            - num_nodes: Number of nodes.
        sf_min_norm (float): Minimum scale factor used for normalization.
        sf_max_norm (float): Maximum scale factor used for normalization.

    Returns:
        dict: A dictionary containing extracted topological features.
              Returns None if TDA fails (e.g., no nodes).
    """
    if data.num_nodes == 0:
        print("Warning: Tree with 0 nodes encountered. Skipping TDA for this tree.")
        return None

    # Un-normalize scale factor (original_sf = norm_sf * (max-min) + min)
    # x[:, 3] is the normalized scale factor
    normalized_sf = data.x[:, 3].numpy()
    original_sf = normalized_sf * (sf_max_norm - sf_min_norm) + sf_min_norm
    
    min_sf_tree = np.min(original_sf) if original_sf.size > 0 else 0.0 # Handle empty original_sf

    st = gd.SimplexTree()

    # Add 0-simplices (nodes)
    for i in range(data.num_nodes):
        st.insert([i], filtration=original_sf[i])

    # Add 1-simplices (edges)
    # Edge_index is [2, num_edges]. Iterate over columns.
    for i in range(data.edge_index.shape[1]):
        u, v = data.edge_index[0, i].item(), data.edge_index[1, i].item()
        filtration_val = max(original_sf[u], original_sf[v])
        st.insert(sorted([u, v]), filtration=filtration_val)

    st.compute_persistence()

    # H0 features
    diag_H0 = st.persistence_intervals_in_dimension(0)
    
    h0_birth_at_min_sf_count = 0
    if diag_H0.size > 0:
        # Ensure min_sf_tree is a scalar for comparison
        h0_birth_at_min_sf_count = np.sum((diag_H0[:, 0] <= min_sf_tree + 1e-9) & (diag_H0[:, 1] > min_sf_tree))

    finite_H0_bars = diag_H0[diag_H0[:, 1] != np.inf]
    h0_avg_persistence_finite = 0.0
    h0_max_persistence_finite = 0.0
    if finite_H0_bars.size > 0:
        persistences_H0 = finite_H0_bars[:, 1] - finite_H0_bars[:, 0]
        h0_avg_persistence_finite = np.mean(persistences_H0) if persistences_H0.size > 0 else 0.0
        h0_max_persistence_finite = np.max(persistences_H0) if persistences_H0.size > 0 else 0.0

    # H1 features
    diag_H1 = st.persistence_intervals_in_dimension(1)
    h1_bar_count = 0
    h1_avg_persistence = 0.0
    h1_max_persistence = 0.0
    if diag_H1.size > 0:
        h1_bar_count = diag_H1.shape[0]
        persistences_H1 = diag_H1[:, 1] - diag_H1[:, 0]
        h1_avg_persistence = np.mean(persistences_H1) if persistences_H1.size > 0 else 0.0
        h1_max_persistence = np.max(persistences_H1) if persistences_H1.size > 0 else 0.0

    features = {
        'num_nodes': data.num_nodes,
        'num_edges': data.edge_index.shape[1],
        'H0_birth_at_min_sf_count': float(h0_birth_at_min_sf_count),
        'H0_avg_persistence_finite': float(h0_avg_persistence_finite),
        'H0_max_persistence_finite': float(h0_max_persistence_finite),
        'H1_bar_count': float(h1_bar_count),
        'H1_avg_persistence': float(h1_avg_persistence),
        'H1_max_persistence': float(h1_max_persistence),
    }

    # Betti numbers at thresholds
    thresholds = [0.25, 0.5, 0.75, 1.0]
    for t in thresholds:
        betti_0_t = 0
        if diag_H0.size > 0:
            betti_0_t = np.sum((diag_H0[:, 0] <= t) & (diag_H0[:, 1] > t))
        features['betti_0_sf_' + str(t)] = float(betti_0_t)

        betti_1_t = 0
        if diag_H1.size > 0:
            betti_1_t = np.sum((diag_H1[:, 0] <= t) & (diag_H1[:, 1] > t))
        features['betti_1_sf_' + str(t)] = float(betti_1_t)
        
    return features, diag_H0, diag_H1, original_sf


def plot_persistence_diagram_custom(diag_H0, diag_H1, tree_idx, timestamp_str, output_dir):
    r"""Plots H0 and H1 persistence diagrams for a tree."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5)) 
    
    if diag_H0.size > 0:
        gd.persistence_graphical_tools.plot_persistence_diagram(diag_H0, axes=axs[0], legend=True)
    axs[0].set_title('H0 Persistence Diagram (Tree ' + str(tree_idx) + ')')
    axs[0].set_xlabel('Birth (Scale Factor)')
    axs[0].set_ylabel('Death (Scale Factor)')
    axs[0].grid(True)
    axs[0].relim()
    axs[0].autoscale_view()

    if diag_H1.size > 0:
        gd.persistence_graphical_tools.plot_persistence_diagram(diag_H1, axes=axs[1], legend=True)
    axs[1].set_title('H1 Persistence Diagram (Tree ' + str(tree_idx) + ')')
    axs[1].set_xlabel('Birth (Scale Factor)')
    axs[1].set_ylabel('Death (Scale Factor)')
    axs[1].grid(True)
    axs[1].relim()
    axs[1].autoscale_view()

    fig.suptitle('Persistence Diagrams for Tree ' + str(tree_idx), fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    
    plot_filename = output_dir / ('persistence_diagram_tree' + str(tree_idx) + '_' + timestamp_str + '.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)
    print("Saved persistence diagram: " + str(plot_filename))
    print("Description: Persistence diagrams for H0 (connected components) and H1 (loops) for merger tree " + str(tree_idx) + ". X-axis is birth time, Y-axis is death time, both in units of scale factor.")


def plot_betti_curves_custom(diag_H0, diag_H1, original_sf_all_nodes, tree_idx, timestamp_str, output_dir):
    r"""Plots Betti curves (B0 and B1) for a tree."""
    
    min_sf_val = np.min(original_sf_all_nodes) if original_sf_all_nodes.size > 0 else 0.0
    max_sf_val = np.max(original_sf_all_nodes) if original_sf_all_nodes.size > 0 else 1.0
    
    if min_sf_val >= max_sf_val: # Ensure a valid range for linspace
        min_sf_val = 0.0
        max_sf_val = 1.0 if max_sf_val <= 0.0 else max_sf_val # if max_sf_val was 0 or less, reset to 1.0
        if min_sf_val >= max_sf_val: # final fallback if max_sf_val was also 0
            max_sf_val = min_sf_val + 0.1


    sf_range = np.linspace(min_sf_val, max_sf_val, 100)
    b0_curve = np.zeros_like(sf_range)
    b1_curve = np.zeros_like(sf_range)

    for i, t in enumerate(sf_range):
        if diag_H0.size > 0:
            b0_curve[i] = np.sum((diag_H0[:, 0] <= t) & (diag_H0[:, 1] > t))
        if diag_H1.size > 0:
            b1_curve[i] = np.sum((diag_H1[:, 0] <= t) & (diag_H1[:, 1] > t))

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))

    axs[0].plot(sf_range, b0_curve, label='Betti_0(t)')
    axs[0].set_title('Betti_0 Curve (Tree ' + str(tree_idx) + ')')
    axs[0].set_xlabel('Scale Factor (t)')
    axs[0].set_ylabel('Number of H0 Components (\u03B2_0)')
    axs[0].grid(True)
    axs[0].legend()
    axs[0].relim()
    axs[0].autoscale_view()

    axs[1].plot(sf_range, b1_curve, label='Betti_1(t)')
    axs[1].set_title('Betti_1 Curve (Tree ' + str(tree_idx) + ')')
    axs[1].set_xlabel('Scale Factor (t)')
    axs[1].set_ylabel('Number of H1 Loops (\u03B2_1)')
    axs[1].grid(True)
    axs[1].legend()
    axs[1].relim()
    axs[1].autoscale_view()
    
    fig.suptitle('Betti Curves for Tree ' + str(tree_idx), fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    
    plot_filename = output_dir / ('betti_curves_tree' + str(tree_idx) + '_' + timestamp_str + '.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)
    print("Saved Betti curves: " + str(plot_filename))
    print("Description: Betti_0 (number of connected components) and Betti_1 (number of loops) as a function of scale factor for merger tree " + str(tree_idx) + ".")


def perform_tda_and_correlation():
    r"""
    Performs TDA on training data, extracts features, calculates correlations,
    and generates plots.
    """
    print("--- Starting Step 2: Topological Data Analysis (TDA) and Correlation ---")
    
    output_dir = Path('data')
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")

    train_data_path = output_dir / 'processed_merger_trees_train.pt'
    norm_params_path = output_dir / 'normalization_parameters.pt'

    if not train_data_path.exists() or not norm_params_path.exists():
        print("Error: Preprocessed data or normalization parameters not found.")
        print("Please run Step 1 (preprocessing) first.")
        return

    print("Loading training data from: " + str(train_data_path))
    train_dataset = torch.load(train_data_path, weights_only=False)
    print("Loading normalization parameters from: " + str(norm_params_path))
    norm_params_loaded = torch.load(norm_params_path, weights_only=False)
    
    node_norm_params = norm_params_loaded['node_norm_params']
    sf_norm_param = node_norm_params[3] 
    sf_min_norm = sf_norm_param['min'].item()
    sf_max_norm = sf_norm_param['max'].item()
    
    print("Number of trees in training set: " + str(len(train_dataset)))
    if not train_dataset:
        print("Training dataset is empty. Cannot perform TDA.")
        return

    all_topo_features = []
    assembly_bias_proxies_list = []  # Renamed to avoid conflict

    print("\n--- Extracting Topological Features from Training Data ---")
    for i, data_tree in enumerate(train_dataset):
        print("Processing tree " + str(i+1) + "/" + str(len(train_dataset)) + "...")
        assembly_bias_proxy = data_tree.y.item() 
        
        topo_results = extract_topological_features(data_tree, sf_min_norm, sf_max_norm)
        
        if topo_results:
            features, diag_H0, diag_H1, original_sf_nodes = topo_results
            features['assembly_bias_proxy'] = assembly_bias_proxy
            all_topo_features.append(features)
            assembly_bias_proxies_list.append(assembly_bias_proxy) 

            if i < 2: # Plot for first 2 trees
                plot_persistence_diagram_custom(diag_H0, diag_H1, i, timestamp_str, output_dir)
                plot_betti_curves_custom(diag_H0, diag_H1, original_sf_nodes, i, timestamp_str, output_dir)
        else:
            print("Skipped TDA for tree " + str(i) + " due to issues (e.g. 0 nodes).")


    if not all_topo_features:
        print("No topological features extracted. Cannot proceed with correlation analysis.")
        return
        
    features_df = pd.DataFrame(all_topo_features)
    
    features_csv_path = output_dir / 'topological_features_train_data.csv'
    features_df.to_csv(features_csv_path, index=False)
    print("\nSaved all extracted topological features and proxies to: " + str(features_csv_path))

    print("\n--- Topological Features for First Few Trees (Validation) ---")
    print(features_df.head(min(3, len(features_df))).to_string())

    print("\n--- Summary Statistics of Topological Features (Training Set) ---")
    # Ensure 'assembly_bias_proxy' column exists before trying to drop it
    cols_to_drop = ['assembly_bias_proxy'] if 'assembly_bias_proxy' in features_df.columns else []
    print(features_df.drop(columns=cols_to_drop).describe().to_string())


    print("\n--- Correlation Analysis: Topological Features vs. Assembly Bias Proxy ---")
    if 'assembly_bias_proxy' not in features_df.columns:
        print("Assembly bias proxy column not found in features DataFrame. Skipping correlation.")
    elif features_df.shape[0] < 2:
        print("Not enough data points (" + str(features_df.shape[0]) + ") to compute correlations.")
    else:
        # Ensure all columns are numeric for correlation
        numeric_features_df = features_df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
        if 'assembly_bias_proxy' not in numeric_features_df.columns:
            print("Assembly bias proxy column became all NaN after numeric conversion. Skipping correlation.")
        else:
            correlation_matrix = numeric_features_df.corr(method='pearson')
            if 'assembly_bias_proxy' in correlation_matrix.columns:
                abp_correlations = correlation_matrix[['assembly_bias_proxy']].sort_values(
                    by='assembly_bias_proxy', ascending=False
                )
                print("Pearson Correlation with Assembly Bias Proxy:")
                print(abp_correlations.to_string())

                plt.figure(figsize=(14, 12)) # Increased figure size
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 7}) # Smaller font for annotations
                plt.title('Correlation Heatmap: Topological Features and Assembly Bias Proxy', fontsize=14)
                plt.xticks(rotation=60, ha='right', fontsize=9) # Rotated more, smaller font
                plt.yticks(fontsize=9) # Smaller font
                plt.tight_layout()
                
                heatmap_filename = output_dir / ('correlation_heatmap_TDA_vs_Bias_0_' + timestamp_str + '.png')
                plt.savefig(heatmap_filename, dpi=300)
                plt.close()
                print("\nSaved correlation heatmap: " + str(heatmap_filename))
                print("Description: Heatmap showing Pearson correlation coefficients between extracted topological features and the assembly bias proxy.")
            else:
                print("Assembly bias proxy column not found in correlation matrix.")


    print("\n--- Step 2: TDA and Correlation Analysis Complete ---")


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    try:
        perform_tda_and_correlation()
    except ImportError as e:
        if 'gudhi' in str(e).lower():
            print("ImportError: Gudhi library not found. Please install it (e.g., pip install gudhi).")
            print("Full error: " + str(e))
        else:
            print("An ImportError occurred: " + str(e))
    except Exception as e:
        print("An unexpected error occurred during TDA and Correlation Analysis:")
        import traceback
        print(traceback.format_exc())
