# filename: codebase/step4_dataset_splitting_regression.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# Configuration
PREPROCESSED_DATA_PATH = 'data/preprocessed_merger_trees.pt'
AGGREGATED_QTT_OUTPUT_DIR = 'data'
AGGREGATED_QTT_FEATURES_PATH_PREFIX = os.path.join(AGGREGATED_QTT_OUTPUT_DIR, 'aggregated_qtt_features_k')
PLOT_OUTPUT_DIR = 'data'  # As per instruction: "ALL files must be saved under the folder data/"
DATABASE_PATH = PLOT_OUTPUT_DIR  # For plot saving convention

K_VALUES = [1, 2, 3]
QTT_RANKS = [2, 3]
FEATURE_NAMES_RAW = ['mass', 'concentration', 'vmax', 'scale_factor']

# Plotting settings
plt.rcParams['text.usetex'] = False
plt.rcParams['figure.dpi'] = 300

# Global plot counter and timestamp
PLOT_COUNTER = 0
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")


def get_plot_filename(plot_name_prefix):
    r"""Generates a unique filename for a plot."""
    global PLOT_COUNTER
    PLOT_COUNTER += 1
    filename = plot_name_prefix + "_" + str(PLOT_COUNTER) + "_" + TIMESTAMP + ".png"
    return os.path.join(DATABASE_PATH, filename)


def create_baseline_features(preprocessed_data_file):
    r"""
    Creates baseline features (mean, max, variance of main branch node features)
    and extracts target values.

    Args:
        preprocessed_data_file (str): Path to the preprocessed_merger_trees.pt file.

    Returns:
        dict: A dictionary containing:
            'X' (torch.Tensor or None): Feature matrix of shape [num_samples, 12].
            'y' (torch.Tensor or None): Target vector of shape [num_samples].
            'feature_names' (list): List of 12 feature names.
            'valid_tree_indices' (list): List of original tree indices included.
        Returns None if data loading fails or no valid trees are found.
    """
    print("Creating baseline features...")
    try:
        trainset = torch.load(preprocessed_data_file, map_location=torch.device('cpu'))
    except Exception as e:
        print("Error loading preprocessed data for baseline: " + str(e))
        return None

    all_baseline_features = []
    all_targets = []
    valid_tree_indices = []

    for tree_idx, tree_data in enumerate(trainset):
        if not (tree_data and hasattr(tree_data, 'x') and tree_data.x is not None and
                hasattr(tree_data, 'mask_main') and tree_data.mask_main is not None and
                hasattr(tree_data, 'num_nodes') and hasattr(tree_data, 'y') and tree_data.y is not None):
            print("  Skipping tree " + str(tree_idx) + " for baseline: missing essential attributes.")
            continue

        node_features = tree_data.x  # Shape: [num_nodes_in_tree, 4]
        mask_main = tree_data.mask_main
        if isinstance(mask_main, np.ndarray):
            mask_main = torch.from_numpy(mask_main).long()
        
        num_nodes_in_tree = tree_data.num_nodes

        # Filter mask_main for valid indices
        valid_mask_indices = (mask_main >= 0) & (mask_main < num_nodes_in_tree)
        main_branch_node_indices = mask_main[valid_mask_indices]

        if main_branch_node_indices.numel() == 0:
            # print(f"  Skipping tree {tree_idx} for baseline: no valid main branch nodes after filtering.")
            continue
            
        main_branch_features = node_features[main_branch_node_indices]  # Shape: [num_main_branch_nodes, 4]

        if main_branch_features.shape[0] == 0:
            # This case should be covered by main_branch_node_indices.numel() == 0
            continue

        # Calculate mean, max, var for each of the 4 features
        # Ensure features are float for var calculation if not already
        main_branch_features_float = main_branch_features.float()
        
        means = torch.mean(main_branch_features_float, dim=0)
        maxs = torch.max(main_branch_features_float, dim=0).values
        variances = torch.var(main_branch_features_float, dim=0, unbiased=False)  # Population variance like numpy.var

        # Handle cases where variance might be NaN (e.g. single node on main branch)
        variances = torch.nan_to_num(variances, nan=0.0)

        tree_baseline_features = torch.cat([means, maxs, variances])  # 4*3 = 12 features
        all_baseline_features.append(tree_baseline_features)
        
        # Assuming y is [1,2] and we predict the first component (final halo mass)
        if tree_data.y.shape == (1,2):
            all_targets.append(tree_data.y[0, 0])
            valid_tree_indices.append(tree_idx)
        else:
            print("  Warning: Tree " + str(tree_idx) + " has unexpected y shape: " + str(tree_data.y.shape) + ". Skipping.")
            all_baseline_features.pop()  # Remove last added features

    if not all_baseline_features:
        print("No valid baseline features could be extracted for any tree.")
        return None

    X_baseline = torch.stack(all_baseline_features)
    y_baseline = torch.tensor(all_targets)

    baseline_feature_names = []
    for stat in ['mean', 'max', 'var']:
        for feat_name in FEATURE_NAMES_RAW:
            baseline_feature_names.append(feat_name + "_" + stat)
    
    print("Baseline features created for " + str(X_baseline.shape[0]) + " trees.")
    print("Baseline feature shape: " + str(X_baseline.shape))
    print("Baseline target shape: " + str(y_baseline.shape))

    return {
        'X': X_baseline, 
        'y': y_baseline, 
        'feature_names': baseline_feature_names,
        'valid_tree_indices': valid_tree_indices
    }


def load_qtt_features(k_val, qtt_rank):
    r"""Loads aggregated QTT features and targets."""
    qtt_file_path = AGGREGATED_QTT_FEATURES_PATH_PREFIX + str(k_val) + '_rank' + str(qtt_rank) + '.pt'
    print("Loading QTT features from: " + qtt_file_path)
    try:
        qtt_data_list = torch.load(qtt_file_path, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print("  QTT feature file not found. Skipping.")
        return None
    except Exception as e:
        print("  Error loading QTT feature file: " + str(e))
        return None

    if not qtt_data_list:
        print("  QTT data list is empty.")
        return None

    all_qtt_features = []
    all_targets = []
    valid_tree_indices_qtt = []

    for item in qtt_data_list:
        # Filter out trees with all-zero QTT vectors (likely placeholders)
        # and ensure target 'y' is valid
        if (torch.abs(item['qtt_feature_vector']).sum() > 1e-9 and 
            item['y'] is not None and item['y'].numel() > 0 and item['y'].shape == (1,2)):
            all_qtt_features.append(item['qtt_feature_vector'])
            all_targets.append(item['y'][0, 0])
            valid_tree_indices_qtt.append(item['tree_idx'])
        
    if not all_qtt_features:
        print("  No valid (non-zero) QTT features found after filtering.")
        return None

    X_qtt = torch.stack(all_qtt_features)
    y_qtt = torch.tensor(all_targets)
    
    qtt_feature_names = ['QTT_feat_' + str(i) for i in range(X_qtt.shape[1])]

    print("  Loaded QTT features for " + str(X_qtt.shape[0]) + " trees (after filtering zero vectors).")
    print("  QTT feature shape: " + str(X_qtt.shape))
    print("  QTT target shape: " + str(y_qtt.shape))
    
    return {
        'X': X_qtt, 
        'y': y_qtt, 
        'feature_names': qtt_feature_names,
        'valid_tree_indices': valid_tree_indices_qtt
    }


def train_evaluate_model(X, y, model_name_suffix, dataset_name, feature_names_list):
    r"""Trains, evaluates a RandomForestRegressor, and generates plots."""
    print("\n--- Training and Evaluating: " + dataset_name + " ---")
    
    if X is None or y is None or X.shape[0] == 0:
        print("  No data available for " + dataset_name + ". Skipping.")
        return None

    num_samples = X.shape[0]
    print("  Number of samples: " + str(num_samples))

    # Convert to numpy for sklearn
    X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
    y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y

    if num_samples <= 5:  # Very small dataset, train and test on all data
        print("  Warning: Dataset size (" + str(num_samples) + ") is very small. Training and evaluating on the full dataset (in-sample performance).")
        X_train, y_train = X_np, y_np
        X_test, y_test = X_np, y_np  # For consistent variable names
        X_val, y_val = X_np, y_np  # Dummy val set
    else:
        # Split: 70% train, 15% validation, 15% test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_np, y_np, test_size=0.15, random_state=42
        )
        # Split train_val into train and validation (0.15 / 0.85 is approx 0.176)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.1765, random_state=42  # 0.15 / (1-0.15)
        )
        print("  Train set size: " + str(X_train.shape[0]))
        print("  Validation set size: " + str(X_val.shape[0]))
        print("  Test set size: " + str(X_test.shape[0]))


    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    print("  Evaluation on Test Set (" + dataset_name + "):")
    print("    Mean Squared Error (MSE): " + str(mse))
    print("    Mean Absolute Error (MAE): " + str(mae))
    print("    R-squared (R²): " + str(r2))

    results = {'mse': mse, 'mae': mae, 'r2': r2, 'model': model, 'name': dataset_name, 'num_samples_test': len(y_test)}

    # --- Plotting ---
    # 1. Predicted vs. True
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred_test, alpha=0.6, edgecolors='w', linewidth=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel("True Values (Final Halo Mass Property)")
    ax.set_ylabel("Predicted Values (Final Halo Mass Property)")
    ax.set_title("Predicted vs. True Values (" + dataset_name + ")")
    ax.grid(True)
    ax.relim()
    ax.autoscale_view()
    plt.tight_layout()
    plot_filename_pvst = get_plot_filename("pred_vs_true_" + model_name_suffix)
    fig.savefig(plot_filename_pvst)
    plt.close(fig)
    print("  Saved Predicted vs. True plot to: " + plot_filename_pvst)

    # 2. Residuals Plot
    residuals = y_test - y_pred_test
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred_test, residuals, alpha=0.6, edgecolors='w', linewidth=0.5)
    ax.hlines(0, y_pred_test.min(), y_pred_test.max(), colors='k', linestyles='--', lw=2)
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals (True - Predicted)")
    ax.set_title("Residuals Plot (" + dataset_name + ")")
    ax.grid(True)
    ax.relim()
    ax.autoscale_view()
    plt.tight_layout()
    plot_filename_res = get_plot_filename("residuals_" + model_name_suffix)
    fig.savefig(plot_filename_res)
    plt.close(fig)
    print("  Saved Residuals plot to: " + plot_filename_res)

    # 3. Feature Importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(feature_names_list) * 0.3)))  # Adjust height
        ax.set_title("Feature Importances (" + dataset_name + ")")
        ax.bar(range(X_train.shape[1]), importances[indices], align="center")
        ax.set_xticks(range(X_train.shape[1]))
        ax.set_xticklabels(np.array(feature_names_list)[indices], rotation=90)
        ax.set_xlim([-1, X_train.shape[1]])
        ax.set_ylabel("Importance")
        ax.grid(True, axis='y')
        ax.relim()
        ax.autoscale_view()
        plt.tight_layout()
        plot_filename_fi = get_plot_filename("feature_importances_" + model_name_suffix)
        fig.savefig(plot_filename_fi)
        plt.close(fig)
        print("  Saved Feature Importances plot to: " + plot_filename_fi)
        results['feature_importances'] = importances
        results['feature_names'] = feature_names_list

    return results


def plot_performance_comparison(all_model_results):
    r"""Plots a comparison of performance metrics for all models."""
    if not all_model_results:
        print("No model results to compare. Skipping performance comparison plot.")
        return

    model_names = [res['name'] for res in all_model_results]
    mse_scores = [res['mse'] for res in all_model_results]
    mae_scores = [res['mae'] for res in all_model_results]
    r2_scores = [res['r2'] for res in all_model_results]

    x = np.arange(len(model_names))
    width = 0.25

    # MSE Plot
    fig, ax = plt.subplots(figsize=(max(10, len(model_names) * 1.5), 6))
    rects1 = ax.bar(x, mse_scores, width, label='MSE')
    ax.set_ylabel('Mean Squared Error (Lower is better)')
    ax.set_title('Model Comparison: MSE')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()
    ax.bar_label(rects1, padding=3, fmt='%.4f')
    ax.grid(True, axis='y')
    ax.relim()
    ax.autoscale_view()
    fig.tight_layout()
    plot_filename_mse_comp = get_plot_filename("comparison_mse")
    fig.savefig(plot_filename_mse_comp)
    plt.close(fig)
    print("\nSaved MSE Comparison plot to: " + plot_filename_mse_comp)

    # MAE Plot
    fig, ax = plt.subplots(figsize=(max(10, len(model_names) * 1.5), 6))
    rects2 = ax.bar(x, mae_scores, width, label='MAE', color='orange')
    ax.set_ylabel('Mean Absolute Error (Lower is better)')
    ax.set_title('Model Comparison: MAE')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()
    ax.bar_label(rects2, padding=3, fmt='%.4f')
    ax.grid(True, axis='y')
    ax.relim()
    ax.autoscale_view()
    fig.tight_layout()
    plot_filename_mae_comp = get_plot_filename("comparison_mae")
    fig.savefig(plot_filename_mae_comp)
    plt.close(fig)
    print("Saved MAE Comparison plot to: " + plot_filename_mae_comp)

    # R2 Plot
    fig, ax = plt.subplots(figsize=(max(10, len(model_names) * 1.5), 6))
    rects3 = ax.bar(x, r2_scores, width, label='R²', color='green')
    ax.set_ylabel('R-squared (Higher is better)')
    ax.set_title('Model Comparison: R² Score')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()
    ax.bar_label(rects3, padding=3, fmt='%.4f')
    ax.grid(True, axis='y')
    ax.relim()
    ax.autoscale_view()
    fig.tight_layout()
    plot_filename_r2_comp = get_plot_filename("comparison_r2")
    fig.savefig(plot_filename_r2_comp)
    plt.close(fig)
    print("Saved R² Comparison plot to: " + plot_filename_r2_comp)


if __name__ == '__main__':
    if not os.path.exists(PLOT_OUTPUT_DIR):
        os.makedirs(PLOT_OUTPUT_DIR)
        print("Created plot output directory: " + PLOT_OUTPUT_DIR)

    all_results_summary = []

    # 1. Baseline Model
    print("\n===== Baseline Model Processing =====")
    baseline_data = create_baseline_features(PREPROCESSED_DATA_PATH)
    if baseline_data and baseline_data['X'] is not None and baseline_data['X'].shape[0] > 0:
        baseline_results = train_evaluate_model(
            baseline_data['X'], 
            baseline_data['y'],
            model_name_suffix="baseline",
            dataset_name="Baseline Features",
            feature_names_list=baseline_data['feature_names']
        )
        if baseline_results:
            all_results_summary.append(baseline_results)
    else:
        print("Could not create or train baseline model due to lack of data.")

    # 2. QTT Models
    print("\n===== QTT Models Processing =====")
    for k in K_VALUES:
        for r in QTT_RANKS:
            model_suffix = "qtt_k" + str(k) + "_r" + str(r)
            dataset_name_str = "QTT (k=" + str(k) + ", rank=" + str(r) + ")"
            
            qtt_data = load_qtt_features(k_val=k, qtt_rank=r)
            
            if qtt_data and qtt_data['X'] is not None and qtt_data['X'].shape[0] > 0:
                qtt_model_results = train_evaluate_model(
                    qtt_data['X'],
                    qtt_data['y'],
                    model_name_suffix=model_suffix,
                    dataset_name=dataset_name_str,
                    feature_names_list=qtt_data['feature_names']
                )
                if qtt_model_results:
                    all_results_summary.append(qtt_model_results)
            else:
                print("Skipping QTT model for k=" + str(k) + ", rank=" + str(r) + " due to no valid/sufficient features.")
    
    # 3. Performance Comparison Plot
    if all_results_summary:
        plot_performance_comparison(all_results_summary)
    else:
        print("\nNo models were successfully trained. Skipping final performance comparison plot.")

    print("\n--- Step 4: Dataset Splitting, Regression Modeling, and Baseline Comparison Complete ---")
