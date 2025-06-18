# filename: codebase/step5_visualization_feature_comparison.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import networkx as nx
import time

# Configuration
PREPROCESSED_DATA_PATH = 'data/preprocessed_merger_trees.pt'
AGGREGATED_QTT_OUTPUT_DIR = 'data'
AGGREGATED_QTT_FEATURES_PATH_PREFIX = os.path.join(AGGREGATED_QTT_OUTPUT_DIR, 'aggregated_qtt_features_k')
SUBGRAPH_COLLECTION_PATH_PREFIX = 'data/k_hop_subgraphs_k'  # For subgraph visualization
PLOT_OUTPUT_DIR = 'data'
DATABASE_PATH = PLOT_OUTPUT_DIR

K_VIS = 1  # k-value for QTT features to visualize
RANK_VIS = 2 # QTT rank for features to visualize
SUBGRAPH_K_VIS = 1 # k-value for subgraph structure to visualize

FEATURE_NAMES_RAW = ['mass', 'concentration', 'vmax', 'scale_factor']

# Plotting settings
plt.rcParams['text.usetex'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'  # Using a common sans-serif font

# Global plot counter and timestamp
PLOT_COUNTER = 0
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")


def get_plot_filename(plot_name_prefix):
    r"""Generates a unique filename for a plot."""
    global PLOT_COUNTER
    PLOT_COUNTER += 1
    filename = plot_name_prefix + "_" + str(PLOT_COUNTER) + "_" + TIMESTAMP + ".png"
    return os.path.join(DATABASE_PATH, filename)


def load_baseline_data(preprocessed_data_file):
    r"""
    Loads preprocessed data and creates baseline features (mean, max, variance of main branch node features)
    and extracts target values. Simplified from Step 4.
    """
    print("Loading data and creating baseline features...")
    try:
        trainset = torch.load(preprocessed_data_file, map_location=torch.device('cpu'))
    except Exception as e:
        print("  Error loading preprocessed data for baseline: " + str(e))
        return None

    all_baseline_features = []
    all_targets = []
    valid_tree_indices = []

    for tree_idx, tree_data in enumerate(trainset):
        if not (tree_data and hasattr(tree_data, 'x') and tree_data.x is not None and
                hasattr(tree_data, 'mask_main') and tree_data.mask_main is not None and
                hasattr(tree_data, 'num_nodes') and hasattr(tree_data, 'y') and tree_data.y is not None):
            continue

        node_features = tree_data.x
        mask_main = tree_data.mask_main
        if isinstance(mask_main, np.ndarray):
            mask_main = torch.from_numpy(mask_main).long()
        
        num_nodes_in_tree = tree_data.num_nodes
        valid_mask_indices = (mask_main >= 0) & (mask_main < num_nodes_in_tree)
        main_branch_node_indices = mask_main[valid_mask_indices]

        if main_branch_node_indices.numel() == 0:
            continue
            
        main_branch_features_tensor = node_features[main_branch_node_indices].float()

        if main_branch_features_tensor.shape[0] == 0:
            continue

        means = torch.mean(main_branch_features_tensor, dim=0)
        maxs = torch.max(main_branch_features_tensor, dim=0).values
        variances = torch.var(main_branch_features_tensor, dim=0, unbiased=False)
        variances = torch.nan_to_num(variances, nan=0.0)

        tree_baseline_features = torch.cat([means, maxs, variances])
        all_baseline_features.append(tree_baseline_features)
        
        if tree_data.y.shape == (1,2):  # Expecting y to be [1,2]
            all_targets.append(tree_data.y[0, 0])  # Target is final halo mass (first component)
            valid_tree_indices.append(tree_idx)
        else:
            all_baseline_features.pop()  # Remove last added features if target is not as expected

    if not all_baseline_features:
        print("  No valid baseline features could be extracted.")
        return None

    X_baseline = torch.stack(all_baseline_features)
    y_baseline = torch.tensor(all_targets)

    baseline_feature_names = []
    for stat in ['mean', 'max', 'var']:
        for feat_name in FEATURE_NAMES_RAW:
            baseline_feature_names.append(feat_name + "_" + stat)
    
    print("  Baseline features loaded/created for " + str(X_baseline.shape[0]) + " trees.")
    return {'X': X_baseline, 'y': y_baseline, 'feature_names': baseline_feature_names}


def load_qtt_data(k_val, qtt_rank):
    r"""Loads aggregated QTT features and targets. Simplified from Step 4."""
    qtt_file_path = AGGREGATED_QTT_FEATURES_PATH_PREFIX + str(k_val) + '_rank' + str(qtt_rank) + '.pt'
    print("Loading QTT features from: " + qtt_file_path)
    try:
        qtt_data_list = torch.load(qtt_file_path, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print("  QTT feature file not found. Returning None.")
        return None
    except Exception as e:
        print("  Error loading QTT feature file: " + str(e) + ". Returning None.")
        return None

    if not qtt_data_list:
        print("  QTT data list is empty. Returning None.")
        return None

    all_qtt_features = []
    all_targets = []

    for item in qtt_data_list:
        if (item.get('qtt_feature_vector') is not None and 
            torch.abs(item['qtt_feature_vector']).sum() > 1e-9 and 
            item.get('y') is not None and item['y'].numel() > 0 and item['y'].shape == (1,2)):
            all_qtt_features.append(item['qtt_feature_vector'])
            all_targets.append(item['y'][0, 0])  # Target is final halo mass
        
    if not all_qtt_features:
        print("  No valid (non-zero) QTT features found after filtering. Returning None.")
        return None

    X_qtt = torch.stack(all_qtt_features)
    y_qtt = torch.tensor(all_targets)
    qtt_feature_names = ['QTT_feat_' + str(i) for i in range(X_qtt.shape[1])]

    print("  Loaded QTT features for " + str(X_qtt.shape[0]) + " trees (after filtering).")
    return {'X': X_qtt, 'y': y_qtt, 'feature_names': qtt_feature_names}


def plot_feature_distributions(features_dict, title_prefix_str, max_features_to_plot=12):
    r"""Plots distributions of features. For small N, plots values directly."""
    if features_dict is None or features_dict['X'].shape[0] == 0:
        print("No features to plot for " + title_prefix_str)
        return

    X = features_dict['X'].cpu().numpy()
    y = features_dict['y'].cpu().numpy()
    feature_names = features_dict['feature_names']
    num_samples, num_features = X.shape

    print("\nPlotting feature distributions for: " + title_prefix_str)
    print("  Number of samples: " + str(num_samples))
    print("  Number of features: " + str(num_features))

    features_to_plot_indices = np.arange(min(num_features, max_features_to_plot))
    
    # Determine number of rows and columns for subplots
    n_cols = 3
    n_rows = (len(features_to_plot_indices) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()  # Flatten to 1D array for easy iteration

    for i, feat_idx in enumerate(features_to_plot_indices):
        ax = axes[i]
        feat_values = X[:, feat_idx]
        
        print("  Feature: " + feature_names[feat_idx] + 
              " (Stats: Min=" + str(np.min(feat_values)) + 
              ", Max=" + str(np.max(feat_values)) + 
              ", Mean=" + str(np.mean(feat_values)) + 
              ", Std=" + str(np.std(feat_values)) + ")")

        if num_samples < 20:  # For very small N, scatter plot individual values
            # Create a jitter array for y-values to prevent overlap
            jitter = np.zeros_like(feat_values)  # No jitter on y for this type of plot
            scatter = ax.scatter(feat_values, jitter, c=y, cmap='viridis', alpha=0.7, s=50)
            ax.set_yticks([])  # Remove y-ticks as they are not meaningful here
            ax.set_xlabel("Feature Value")
        else:  # For larger N, use histogram
            scatter = ax.hist(feat_values, bins=10, color='skyblue', edgecolor='black')
            ax.set_xlabel("Feature Value")
            ax.set_ylabel("Frequency")

        ax.set_title(feature_names[feat_idx], fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.7)

    # Add a colorbar for the target variable if y is used for coloring
    if num_samples < 20 and y is not None:
        fig.colorbar(scatter, ax=axes[:len(features_to_plot_indices)], label='Target (Halo Mass Property)', aspect=30*n_rows)

    # Hide any unused subplots
    for j in range(len(features_to_plot_indices), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Feature Distributions: " + title_prefix_str + " (First " + str(len(features_to_plot_indices)) + " features)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to make space for suptitle
    
    plot_filename = get_plot_filename("feature_dist_" + title_prefix_str.lower().replace(" ", "_"))
    fig.savefig(plot_filename)
    plt.close(fig)
    print("  Saved feature distribution plot to: " + plot_filename)


def plot_pca_reduction(features_dict, title_prefix_str):
    r"""Performs PCA and plots the 2D reduced representation."""
    if features_dict is None or features_dict['X'].shape[0] == 0:
        print("No features for PCA for " + title_prefix_str)
        return

    X = features_dict['X'].cpu().numpy()
    y = features_dict['y'].cpu().numpy()
    num_samples = X.shape[0]

    print("\nPlotting PCA reduction for: " + title_prefix_str)
    print("  Number of samples for PCA: " + str(num_samples))

    if num_samples < 2:
        print("  Not enough samples for PCA. Skipping.")
        return
    
    # n_components cannot be larger than min(n_samples, n_features)
    # For visualization, we want 2 components. If n_samples < 2, PCA(2) fails.
    # If n_samples == 2, PCA can give 1 component.
    n_pca_components = min(2, num_samples - 1 if num_samples > 1 else 1)
    if n_pca_components < 1:
        print("  Cannot perform PCA with less than 1 component (num_samples=" + str(num_samples) + "). Skipping.")
        return

    pca = PCA(n_components=n_pca_components)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 7))
    if n_pca_components == 1:
        # Plot as a 1D scatter plot if only one component
        jitter = np.zeros_like(y)  # Plot along a line
        scatter = ax.scatter(X_pca[:, 0], jitter, c=y, cmap='viridis', alpha=0.8, s=70)
        ax.set_xlabel("Principal Component 1 (Explained Var: " + "{:.2f}".format(pca.explained_variance_ratio_[0]) + ")")
        ax.set_yticks([])
        ax.set_title("1D PCA of " + title_prefix_str + " (Colored by Halo Mass Property)", fontsize=14)
    else:  # n_pca_components == 2
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8, s=70)
        ax.set_xlabel("Principal Component 1 (Explained Var: " + "{:.2f}".format(pca.explained_variance_ratio_[0]) + ")")
        ax.set_ylabel("Principal Component 2 (Explained Var: " + "{:.2f}".format(pca.explained_variance_ratio_[1]) + ")")
        ax.set_title("2D PCA of " + title_prefix_str + " (Colored by Halo Mass Property)", fontsize=14)

    fig.colorbar(scatter, ax=ax, label='Target (Halo Mass Property)')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.relim()
    ax.autoscale_view()
    plt.tight_layout()
    
    plot_filename = get_plot_filename("pca_" + title_prefix_str.lower().replace(" ", "_"))
    fig.savefig(plot_filename)
    plt.close(fig)
    print("  Saved PCA plot to: " + plot_filename)
    print("  Explained variance ratio: " + str(pca.explained_variance_ratio_))


def visualize_selected_subgraph(subgraph_collection_path, subgraph_idx_to_plot=0):
    r"""Visualizes a selected k-hop subgraph, coloring nodes by a feature."""
    print("\nVisualizing selected subgraph...")
    print("  Loading subgraph data from: " + subgraph_collection_path)
    try:
        subgraph_list = torch.load(subgraph_collection_path, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print("  Subgraph data file not found. Skipping subgraph visualization.")
        return
    except Exception as e:
        print("  Error loading subgraph data: " + str(e) + ". Skipping.")
        return

    if not subgraph_list or subgraph_idx_to_plot >= len(subgraph_list):
        print("  Subgraph list is empty or index is out of bounds. Skipping.")
        return

    subgraph_data = subgraph_list[subgraph_idx_to_plot]
    
    k_val = subgraph_data.get('k_value', 'Unknown_k')
    tree_id = subgraph_data.get('tree_id', 'Unknown_tree')
    original_mb_node_idx = subgraph_data.get('main_branch_node_original_idx', 'N/A')

    print("  Visualizing subgraph " + str(subgraph_idx_to_plot) + " from tree_id " + str(tree_id) + 
          " (k=" + str(k_val) + ", original main branch node index: " + str(original_mb_node_idx) + ")")

    # Node features (already preprocessed and padded)
    # Shape: [fixed_num_nodes_for_qtt, num_features]
    node_features_padded = subgraph_data['padded_feature_matrix']
    
    # Actual number of nodes in this specific subgraph before padding
    num_actual_nodes = subgraph_data['num_nodes_in_original_subgraph']
    
    if num_actual_nodes == 0:
        print("  Selected subgraph has 0 actual nodes. Skipping visualization.")
        return

    # Use only the actual nodes' features
    node_features_actual = node_features_padded[:num_actual_nodes, :]
    
    # Edge index (relabelled for the subgraph)
    # Shape: [2, num_edges_in_subgraph]
    edge_index = subgraph_data['subgraph_edge_index']

    # Create graph using networkx
    G = nx.Graph()
    G.add_nodes_from(range(num_actual_nodes))
    
    # Add edges. Edge_index is for directed graph, convert to list of tuples for undirected
    # Also, ensure edges are within the num_actual_nodes range
    valid_edges = []
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        if u < num_actual_nodes and v < num_actual_nodes:
            valid_edges.append((u, v))
    G.add_edges_from(valid_edges)

    # Node colors based on the first feature (e.g., standardized log-mass)
    # Ensure feature values are scalar for coloring
    node_colors = node_features_actual[:, 0].cpu().numpy().flatten()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)  # For reproducible layout
    
    # Draw nodes
    nodes_plot = nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.viridis, 
                                        node_size=500, alpha=0.9, ax=ax)
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    ax.set_title("Visualization of Subgraph " + str(subgraph_idx_to_plot) + " (k=" + str(k_val) + ")\n" +
                 "Tree ID: " + str(tree_id) + ", Original Main Branch Node: " + str(original_mb_node_idx) + "\n" +
                 "Nodes colored by 1st feature (e.g., standardized log-mass)", fontsize=12)
    plt.colorbar(nodes_plot, ax=ax, label="Standardized Feature 1 Value", shrink=0.8)
    ax.axis('off')  # Turn off axis
    plt.tight_layout()

    plot_filename = get_plot_filename("subgraph_vis_k" + str(k_val) + "_idx" + str(subgraph_idx_to_plot))
    fig.savefig(plot_filename)
    plt.close(fig)
    print("  Saved subgraph visualization to: " + plot_filename)


if __name__ == '__main__':
    if not os.path.exists(PLOT_OUTPUT_DIR):
        os.makedirs(PLOT_OUTPUT_DIR)
        print("Created plot output directory: " + PLOT_OUTPUT_DIR)

    print("--- Starting Step 5: Visualization and Feature Space Comparison ---")

    # 1. Load Baseline Features
    baseline_features_data = load_baseline_data(PREPROCESSED_DATA_PATH)

    # 2. Load QTT Features (for a selected k and rank)
    qtt_features_data = load_qtt_data(k_val=K_VIS, qtt_rank=RANK_VIS)

    # 3. Plot Feature Distributions
    if baseline_features_data:
        plot_feature_distributions(baseline_features_data, "Baseline Features")
    if qtt_features_data:
        plot_feature_distributions(qtt_features_data, "QTT Features (k=" + str(K_VIS) + ", r=" + str(RANK_VIS) + ")")

    # 4. Plot PCA Dimensionality Reduction
    if baseline_features_data:
        plot_pca_reduction(baseline_features_data, "Baseline Features")
    if qtt_features_data:
        plot_pca_reduction(qtt_features_data, "QTT Features (k=" + str(K_VIS) + ", r=" + str(RANK_VIS) + ")")

    # 5. Visualize a Selected Subgraph
    # Path to the k-hop subgraph data file (e.g., for k=1)
    subgraph_file_to_vis = SUBGRAPH_COLLECTION_PATH_PREFIX + str(SUBGRAPH_K_VIS) + '.pt'
    visualize_selected_subgraph(subgraph_file_to_vis, subgraph_idx_to_plot=0)
                                     
    print("\n--- Step 5: Visualization and Feature Space Comparison Complete ---")