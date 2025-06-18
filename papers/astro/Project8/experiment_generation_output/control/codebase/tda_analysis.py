# filename: codebase/tda_analysis.py
import torch
import gudhi as gd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import os

# Ensure reproducibility for any random operations if added later
torch.manual_seed(42)
np.random.seed(42)

# Matplotlib configuration
plt.rcParams['text.usetex'] = False
plt.rcParams['figure.dpi'] = 300


def compute_tda_features_for_tree(data_obj):
    r"""
    Constructs a simplicial complex for a single merger tree, computes persistent
    homology, and extracts topological features.

    Args:
        data_obj (torch_geometric.data.Data): A single merger tree object.
            It must contain:
            - x: Node features, where x[:, 3] is the normalized scale factor.
            - edge_index: Edge connectivity.

    Returns:
        dict: A dictionary containing extracted TDA features.
              Returns None if the tree has no nodes.
    """
    if data_obj.num_nodes == 0:
        return None

    node_sfs = data_obj.x[:, 3].numpy()  # Normalized scale factors
    edge_index = data_obj.edge_index.numpy()

    st = gd.SimplexTree()

    # Add 0-simplices (nodes) with their scale factor as filtration value
    min_sf_in_tree = np.inf
    if len(node_sfs) > 0:
        min_sf_in_tree = np.min(node_sfs)
        
    for i in range(data_obj.num_nodes):
        st.insert([i], filtration=node_sfs[i])

    # Add 1-simplices (edges) with max scale factor of their nodes
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        # Ensure u and v are within node bounds, though typically they should be
        if u < data_obj.num_nodes and v < data_obj.num_nodes:
            filtration_val = max(node_sfs[u], node_sfs[v])
            st.insert([u, v], filtration=filtration_val)
        else:
            print("Warning: Edge (" + str(u) + "," + str(v) + ") has out-of-bounds node index for num_nodes=" + str(data_obj.num_nodes))

    st.make_filtration_non_decreasing()  # Important for gudhi
    st.compute_persistence(persistence_dim_max=True)  # Compute H0, H1

    # H0 features
    h0_intervals = st.persistence_intervals_in_dimension(0)
    num_H0_bars_born_at_min_sf = 0
    h0_finite_persistence_values = []
    
    if len(h0_intervals) > 0:
        for birth, death in h0_intervals:
            if np.isclose(birth, min_sf_in_tree):
                 num_H0_bars_born_at_min_sf += 1
            if not np.isinf(death):
                h0_finite_persistence_values.append(death - birth)
    
    avg_pers_H0_finite = np.mean(h0_finite_persistence_values) if h0_finite_persistence_values else 0
    max_pers_H0_finite = np.max(h0_finite_persistence_values) if h0_finite_persistence_values else 0

    # H1 features
    h1_intervals = st.persistence_intervals_in_dimension(1)
    h1_persistence_values = []
    if len(h1_intervals) > 0:
        for birth, death in h1_intervals:
            if not np.isinf(death):  # H1 deaths are usually finite for merger trees
                h1_persistence_values.append(death - birth)

    avg_pers_H1 = np.mean(h1_persistence_values) if h1_persistence_values else 0
    max_pers_H1 = np.max(h1_persistence_values) if h1_persistence_values else 0

    # Betti numbers at thresholds
    thresholds = [0.25, 0.5, 0.75, 1.0]
    betti_features = {}

    for t in thresholds:
        b0_t = 0
        if len(h0_intervals) > 0:
            for birth, death in h0_intervals:
                if birth <= t < death:  # death can be inf
                    b0_t += 1
        betti_features['betti_0_sf_' + str(t)] = b0_t

        b1_t = 0
        if len(h1_intervals) > 0:
            for birth, death in h1_intervals:
                if birth <= t < death:  # death can be inf (though rare for H1)
                    b1_t += 1
        betti_features['betti_1_sf_' + str(t)] = b1_t
        
    features = {
        'num_H0_bars_born_at_min_sf': num_H0_bars_born_at_min_sf,
        'avg_pers_H0_finite': avg_pers_H0_finite,
        'max_pers_H0_finite': max_pers_H0_finite,
        'avg_pers_H1': avg_pers_H1,
        'max_pers_H1': max_pers_H1,
    }
    features.update(betti_features)
    
    # Store persistence intervals for potential plotting of the first tree
    if not hasattr(compute_tda_features_for_tree, 'first_tree_persistence_intervals'):
        compute_tda_features_for_tree.first_tree_persistence_intervals = {
            'h0': h0_intervals,
            'h1': h1_intervals,
            'min_sf': min_sf_in_tree,
            'max_sf': np.max(node_sfs) if len(node_sfs) > 0 else 1.0
        }
        
    # Store Betti numbers over fine scale factor range for average Betti curve plot
    sf_range_for_betti = np.linspace(min_sf_in_tree if not np.isinf(min_sf_in_tree) else 0, 
                                     np.max(node_sfs) if len(node_sfs) > 0 else 1.0, 
                                     50)
    betti_0_curve = []
    betti_1_curve = []
    for t_val in sf_range_for_betti:
        b0_val = 0
        if len(h0_intervals) > 0:
            for birth, death in h0_intervals:
                if birth <= t_val < death:
                    b0_val += 1
        betti_0_curve.append(b0_val)
        
        b1_val = 0
        if len(h1_intervals) > 0:
            for birth, death in h1_intervals:
                if birth <= t_val < death:
                    b1_val += 1
        betti_1_curve.append(b1_val)
    
    features['betti_0_curve_points'] = (sf_range_for_betti, betti_0_curve)
    features['betti_1_curve_points'] = (sf_range_for_betti, betti_1_curve)

    return features


def run_tda_analysis():
    r"""
    Main function to run the TDA pipeline: load data, compute features,
    calculate correlations, and generate plots.
    """
    start_time_overall = time.time()
    plot_counter = 1
    
    # --- Configuration ---
    base_dir = Path('.')
    data_dir = base_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)  # Ensure data directory exists

    train_data_path = data_dir / 'processed_merger_trees_train.pt'
    
    print("--- Topological Data Analysis (TDA) ---")

    # --- 1. Load Data ---
    print("\nLoading preprocessed training data from: " + str(train_data_path))
    if not train_data_path.exists():
        print("Error: Training data file not found at " + str(train_data_path))
        print("Please ensure Step 1 (preprocessing) was run successfully.")
        return
    try:
        train_dataset = torch.load(train_data_path)
    except Exception as e:
        print("Error loading training dataset: " + str(e))
        return

    if not train_dataset:
        print("Training dataset is empty. Cannot perform TDA.")
        return
    print("Loaded " + str(len(train_dataset)) + " training merger trees.")

    # --- 2. Compute TDA Features for each tree ---
    print("\nComputing TDA features for each tree...")
    all_tda_features = []
    assembly_bias_proxies = []
    
    # For average Betti curves
    all_betti_0_curves = []
    all_betti_1_curves = []
    common_sf_range = np.linspace(0, 1, 50)  # Common scale factor range for averaging

    for i, tree_data in enumerate(train_dataset):
        features = compute_tda_features_for_tree(tree_data)
        if features:
            all_tda_features.append(features)
            assembly_bias_proxies.append(tree_data.y.item())  # y is assembly bias proxy

            # Interpolate Betti curves to common_sf_range for averaging
            sf_points, b0_points = features['betti_0_curve_points']
            if len(sf_points) > 1:  # Need at least 2 points to interpolate
                interp_b0 = np.interp(common_sf_range, sf_points, b0_points, left=b0_points[0], right=b0_points[-1])
                all_betti_0_curves.append(interp_b0)

            sf_points, b1_points = features['betti_1_curve_points']
            if len(sf_points) > 1:
                interp_b1 = np.interp(common_sf_range, sf_points, b1_points, left=b1_points[0], right=b1_points[-1])
                all_betti_1_curves.append(interp_b1)
        
        # Remove curve points from features dict before creating DataFrame
        if features:
            del features['betti_0_curve_points']
            del features['betti_1_curve_points']

    if not all_tda_features:
        print("No TDA features could be extracted. Exiting.")
        return

    tda_df = pd.DataFrame(all_tda_features)
    tda_df['assembly_bias_proxy'] = assembly_bias_proxies

    print("\n--- TDA Feature Summary (First 5 trees) ---")
    print(tda_df.head().to_string())

    # --- 3. Correlation Analysis ---
    print("\n--- Correlation Analysis ---")
    correlation_matrix = tda_df.corr(method='pearson')
    print("Pearson Correlation Matrix with Assembly Bias Proxy:")
    print(correlation_matrix[['assembly_bias_proxy']].sort_values(by='assembly_bias_proxy', ascending=False).to_string())

    # --- 4. Generate Plots ---
    print("\n--- Generating Plots ---")
    current_timestamp = int(time.time())

    # Plot 1: Persistence Diagram for the first tree
    if hasattr(compute_tda_features_for_tree, 'first_tree_persistence_intervals'):
        persistence_info = compute_tda_features_for_tree.first_tree_persistence_intervals
        fig, ax = plt.subplots(figsize=(7, 7))
        
        # Manually add labels for legend if needed, or rely on gudhi's default if it provides them
        # For more control, plot H0 and H1 separately if legend becomes an issue
        h0_plot_data = [(birth, death) for birth, death in persistence_info['h0']]
        h1_plot_data = [(birth, death) for birth, death in persistence_info['h1']]

        gd.plot_persistence_diagram(h0_plot_data, axes=ax, legend=True)
        gd.plot_persistence_diagram(h1_plot_data, axes=ax, legend=True)
        
        # Add manual legend entries if gudhi's plot doesn't create them as desired
        if h0_plot_data:
             ax.plot([], [], color='C0', label='H0')
        if h1_plot_data:
             ax.plot([], [], color='C1', label='H1')
        ax.legend()

        ax.set_title("Persistence Diagram (First Training Tree)")
        ax.set_xlabel("Birth (Scale Factor)")
        ax.set_ylabel("Death (Scale Factor)")
        plt.tight_layout()
        
        plot_filename = data_dir / ("persistence_diagram_" + str(plot_counter) + "_" + str(current_timestamp) + ".png")
        plt.savefig(plot_filename)
        print("Saved persistence diagram to: " + str(plot_filename))
        print("Description: Persistence diagram for H0 (connected components) and H1 (loops) of the first merger tree in the training set. X-axis is birth scale factor, Y-axis is death scale factor.")
        plt.close(fig)
        plot_counter += 1
    else:
        print("Could not generate persistence diagram (no tree processed or first tree had no features).")

    # Plot 2: Average Betti Curves
    if all_betti_0_curves and all_betti_1_curves:
        avg_betti_0 = np.mean(np.array(all_betti_0_curves), axis=0)
        avg_betti_1 = np.mean(np.array(all_betti_1_curves), axis=0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(common_sf_range, avg_betti_0, label='Average beta_0 (Connected Components)', color='blue')
        ax.plot(common_sf_range, avg_betti_1, label='Average beta_1 (Loops)', color='red')
        
        ax.set_xlabel("Normalized Scale Factor (t)")
        ax.set_ylabel("Average Betti Number")
        ax.set_title("Average Betti Curves over Training Set")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.relim()
        ax.autoscale_view()
        plt.tight_layout()

        plot_filename = data_dir / ("avg_betti_curves_" + str(plot_counter) + "_" + str(current_timestamp) + ".png")
        plt.savefig(plot_filename)
        print("Saved average Betti curves plot to: " + str(plot_filename))
        print("Description: Average Betti numbers (beta_0 and beta_1) as a function of normalized scale factor, averaged over all training merger trees.")
        plt.close(fig)
        plot_counter += 1
    else:
        print("Could not generate average Betti curves (insufficient data).")

    # Plot 3: Correlation Heatmap
    if not correlation_matrix.empty:
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax, annot_kws={"size":8})
        ax.set_title("TDA Feature Correlation Heatmap (with Assembly Bias Proxy)")
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()

        plot_filename = data_dir / ("tda_feature_correlation_heatmap_" + str(plot_counter) + "_" + str(current_timestamp) + ".png")
        plt.savefig(plot_filename)
        print("Saved TDA feature correlation heatmap to: " + str(plot_filename))
        print("Description: Heatmap showing Pearson correlation coefficients between extracted TDA features and the assembly bias proxy.")
        plt.close(fig)
        plot_counter += 1
    else:
        print("Could not generate correlation heatmap (correlation matrix is empty).")
        
    end_time_overall = time.time()
    print("\n--- TDA Analysis Complete ---")
    print("Total time taken: " + str(round(end_time_overall - start_time_overall, 2)) + " seconds")


if __name__ == '__main__':
    run_tda_analysis()