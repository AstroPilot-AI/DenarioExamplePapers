# filename: codebase/merger_tree_plots.py
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import joblib
from sklearn.impute import SimpleImputer  # For handling potential NaNs in loaded features for plotting

# --- Configuration ---
DATABASE_PATH = "data/"
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
plt.rcParams['text.usetex'] = False  # Ensure LaTeX is off
os.makedirs(DATABASE_PATH, exist_ok=True)

# Define feature names for clarity in plots
X_ENGINEERED_FEATURE_NAMES = [
    'mean_delta_sf', 'var_delta_sf', 'mean_log_mass_ratio', 'var_log_mass_ratio',  # Edge (4)
    'lap_eig_mean', 'lap_eig_std', 'lap_eig_skew', 'lap_eig_kurt', 'lap_eig_sum_smallest',  # Laplacian (5)
    'diff_mean_eig1', 'diff_mean_eig2', 'diff_mean_eig3', 'diff_mean_eig4', 'diff_mean_eig5',  # Diffusion Mean (5)
    'diff_max_eig1', 'diff_max_eig2', 'diff_max_eig3', 'diff_max_eig4', 'diff_max_eig5',  # Diffusion Max (5)
    'diff_min_eig1', 'diff_min_eig2', 'diff_min_eig3', 'diff_min_eig4', 'diff_min_eig5'   # Diffusion Min (5)
]
# Indices for engineered features
EDGE_FEAT_INDICES = slice(0, 4)
LAPLACIAN_FEAT_INDICES = slice(4, 9)
DIFFUSION_MEAN_FEAT_INDICES = slice(9, 14)
DIFFUSION_MAX_FEAT_INDICES = slice(14, 19)
DIFFUSION_MIN_FEAT_INDICES = slice(19, 24)


X_BASELINE_FEATURE_NAMES = [
    'logM_mean', 'logC_mean', 'logV_mean', 'SF_mean',
    'logM_std', 'logC_std', 'logV_std', 'SF_std',
    'logM_min', 'logC_min', 'logV_min', 'SF_min',
    'logM_max', 'logC_max', 'logV_max', 'SF_max'
]

TARGET_VARIABLES = ['Omega_m', 'sigma_8']

# --- Plotting Functions ---

def plot_engineered_feature_distributions(X_engineered, feature_names, plot_num_start=1):
    """Plots distributions of engineered features."""
    print("Plotting distributions of engineered features...")
    
    # Impute NaNs for plotting if any (though they should be handled per feature)
    imputer = SimpleImputer(strategy='mean')
    X_engineered_imputed = imputer.fit_transform(X_engineered)

    # Plot 1: Edge Features
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
    axes1 = axes1.ravel()
    for i in range(EDGE_FEAT_INDICES.stop - EDGE_FEAT_INDICES.start):
        idx = EDGE_FEAT_INDICES.start + i
        feat_data = X_engineered[:, idx]
        feat_data_no_nan = feat_data[~np.isnan(feat_data)]
        axes1[i].hist(feat_data_no_nan, bins=50, density=True, alpha=0.7, color='skyblue')
        axes1[i].set_title(feature_names[idx])
        axes1[i].set_xlabel("Value")
        axes1[i].set_ylabel("Density")
        axes1[i].grid(True, linestyle='--', alpha=0.7)
    fig1.suptitle("Distributions of Edge Features", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = os.path.join(DATABASE_PATH, "engineered_feature_dist_edge_" + str(plot_num_start) + "_" + TIMESTAMP + ".png")
    plt.savefig(filename, dpi=300)
    print("Saved plot: " + filename + " - Distributions of 4 edge-based engineered features.")
    plt.close(fig1)

    # Plot 2: Laplacian Spectral Features
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    axes2 = axes2.ravel()
    for i in range(LAPLACIAN_FEAT_INDICES.stop - LAPLACIAN_FEAT_INDICES.start):
        idx = LAPLACIAN_FEAT_INDICES.start + i
        feat_data = X_engineered[:, idx]
        feat_data_no_nan = feat_data[~np.isnan(feat_data)]
        axes2[i].hist(feat_data_no_nan, bins=50, density=True, alpha=0.7, color='lightcoral')
        axes2[i].set_title(feature_names[idx])
        axes2[i].set_xlabel("Value")
        axes2[i].set_ylabel("Density")
        axes2[i].grid(True, linestyle='--', alpha=0.7)
    if len(axes2) > (LAPLACIAN_FEAT_INDICES.stop - LAPLACIAN_FEAT_INDICES.start):  # Hide unused subplot
        axes2[-1].axis('off')
    fig2.suptitle("Distributions of Laplacian Spectral Features", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = os.path.join(DATABASE_PATH, "engineered_feature_dist_laplacian_" + str(plot_num_start + 1) + "_" + TIMESTAMP + ".png")
    plt.savefig(filename, dpi=300)
    print("Saved plot: " + filename + " - Distributions of 5 Laplacian spectral features.")
    plt.close(fig2)

    # Plot 3a: Diffusion Mean Features
    fig3a, axes3a = plt.subplots(2, 3, figsize=(18, 10))
    axes3a = axes3a.ravel()
    for i in range(DIFFUSION_MEAN_FEAT_INDICES.stop - DIFFUSION_MEAN_FEAT_INDICES.start):
        idx = DIFFUSION_MEAN_FEAT_INDICES.start + i
        feat_data = X_engineered[:, idx]
        feat_data_no_nan = feat_data[~np.isnan(feat_data)]
        axes3a[i].hist(feat_data_no_nan, bins=50, density=True, alpha=0.7, color='mediumseagreen')
        axes3a[i].set_title(feature_names[idx])
        axes3a[i].set_xlabel("Value")
        axes3a[i].set_ylabel("Density")
        axes3a[i].grid(True, linestyle='--', alpha=0.7)
    if len(axes3a) > (DIFFUSION_MEAN_FEAT_INDICES.stop - DIFFUSION_MEAN_FEAT_INDICES.start):
        axes3a[-1].axis('off')
    fig3a.suptitle("Distributions of Diffusion Map Mean-Pooled Embeddings", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = os.path.join(DATABASE_PATH, "engineered_feature_dist_diff_mean_" + str(plot_num_start + 2) + "_" + TIMESTAMP + ".png")
    plt.savefig(filename, dpi=300)
    print("Saved plot: " + filename + " - Distributions of 5 diffusion map mean-pooled embedding features.")
    plt.close(fig3a)
    
    # Plot 3b: Diffusion Max Features
    fig3b, axes3b = plt.subplots(2, 3, figsize=(18, 10))
    axes3b = axes3b.ravel()
    for i in range(DIFFUSION_MAX_FEAT_INDICES.stop - DIFFUSION_MAX_FEAT_INDICES.start):
        idx = DIFFUSION_MAX_FEAT_INDICES.start + i
        feat_data = X_engineered[:, idx]
        feat_data_no_nan = feat_data[~np.isnan(feat_data)]
        axes3b[i].hist(feat_data_no_nan, bins=50, density=True, alpha=0.7, color='goldenrod')
        axes3b[i].set_title(feature_names[idx])
        axes3b[i].set_xlabel("Value")
        axes3b[i].set_ylabel("Density")
        axes3b[i].grid(True, linestyle='--', alpha=0.7)
    if len(axes3b) > (DIFFUSION_MAX_FEAT_INDICES.stop - DIFFUSION_MAX_FEAT_INDICES.start):
        axes3b[-1].axis('off')
    fig3b.suptitle("Distributions of Diffusion Map Max-Pooled Embeddings", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = os.path.join(DATABASE_PATH, "engineered_feature_dist_diff_max_" + str(plot_num_start + 3) + "_" + TIMESTAMP + ".png")
    plt.savefig(filename, dpi=300)
    print("Saved plot: " + filename + " - Distributions of 5 diffusion map max-pooled embedding features.")
    plt.close(fig3b)

    # Plot 3c: Diffusion Min Features
    fig3c, axes3c = plt.subplots(2, 3, figsize=(18, 10))
    axes3c = axes3c.ravel()
    for i in range(DIFFUSION_MIN_FEAT_INDICES.stop - DIFFUSION_MIN_FEAT_INDICES.start):
        idx = DIFFUSION_MIN_FEAT_INDICES.start + i
        feat_data = X_engineered[:, idx]
        feat_data_no_nan = feat_data[~np.isnan(feat_data)]
        axes3c[i].hist(feat_data_no_nan, bins=50, density=True, alpha=0.7, color='slateblue')
        axes3c[i].set_title(feature_names[idx])
        axes3c[i].set_xlabel("Value")
        axes3c[i].set_ylabel("Density")
        axes3c[i].grid(True, linestyle='--', alpha=0.7)
    if len(axes3c) > (DIFFUSION_MIN_FEAT_INDICES.stop - DIFFUSION_MIN_FEAT_INDICES.start):
        axes3c[-1].axis('off')
    fig3c.suptitle("Distributions of Diffusion Map Min-Pooled Embeddings", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = os.path.join(DATABASE_PATH, "engineered_feature_dist_diff_min_" + str(plot_num_start + 4) + "_" + TIMESTAMP + ".png")
    plt.savefig(filename, dpi=300)
    print("Saved plot: " + filename + " - Distributions of 5 diffusion map min-pooled embedding features.")
    plt.close(fig3c)
    return plot_num_start + 5


def plot_pca_projection(X_pca, y_targets, plot_num_start=1):
    """Plots projection of graphs in the first two principal components."""
    print("Plotting PCA projection...")
    if X_pca.shape[1] < 2:
        print("  Skipping PCA projection plot as number of components is less than 2.")
        return plot_num_start
        
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Color by Omega_m
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_targets[:, 0], cmap='viridis', alpha=0.7, s=15)
    axes[0].set_title('PCA Projection (Colored by Omega_m)')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    cbar1 = fig.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('Omega_m')
    
    # Color by sigma_8
    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_targets[:, 1], cmap='plasma', alpha=0.7, s=15)
    axes[1].set_title('PCA Projection (Colored by sigma_8)')
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 2')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    cbar2 = fig.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('sigma_8')
    
    plt.tight_layout()
    filename = os.path.join(DATABASE_PATH, "pca_projection_plot_" + str(plot_num_start) + "_" + TIMESTAMP + ".png")
    plt.savefig(filename, dpi=300)
    print("Saved plot: " + filename + " - PCA projection (PC1 vs PC2) colored by Omega_m and sigma_8.")
    plt.close(fig)
    return plot_num_start + 1


def plot_feature_importances(target_name, pca_rfr_model, pca_gbr_model, baseline_rfr_model, baseline_gbr_model,
                             pca_feature_names, baseline_feature_names, plot_num_start=1):
    """Plots feature importances for tree-based models."""
    print("Plotting feature importances for " + target_name + "...")
    
    models_info = {
        "PCA RFR": (pca_rfr_model, pca_feature_names),
        "PCA GBR": (pca_gbr_model, pca_feature_names),
        "Baseline RFR": (baseline_rfr_model, baseline_feature_names),
        "Baseline GBR": (baseline_gbr_model, baseline_feature_names)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.ravel()
    
    for i, (model_label, (model, feat_names)) in enumerate(models_info.items()):
        if model is None:
            axes[i].text(0.5, 0.5, "Model not available", ha='center', va='center', fontsize=12, color='red')
            axes[i].set_title(model_label + " - Importances")
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            continue

        importances = model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        
        axes[i].bar(range(len(feat_names)), importances[sorted_indices], align='center', color='cornflowerblue')
        axes[i].set_xticks(range(len(feat_names)))
        axes[i].set_xticklabels(np.array(feat_names)[sorted_indices], rotation=90, ha='right')
        axes[i].set_title(model_label + " - Importances")
        axes[i].set_ylabel("Importance")
        axes[i].grid(True, axis='y', linestyle='--', alpha=0.7)

    fig.suptitle("Feature Importances for " + target_name + " Prediction", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust rect to make space for suptitle and x-labels
    filename = os.path.join(DATABASE_PATH, "feature_importances_" + target_name + "_" + str(plot_num_start) + "_" + TIMESTAMP + ".png")
    plt.savefig(filename, dpi=300)
    print("Saved plot: " + filename + " - Feature importances for " + target_name + ".")
    plt.close(fig)
    return plot_num_start + 1


def plot_predicted_vs_true(target_name, results_data, plot_num_start=1):
    """Plots predicted vs. true values for all models for a given target."""
    print("Plotting Predicted vs. True for " + target_name + "...")
    
    pca_results = results_data.get('pca_engineered', {}).item() if 'pca_engineered' in results_data else {}
    baseline_results = results_data.get('baseline_aggregated', {}).item() if 'baseline_aggregated' in results_data else {}
    gcn_results = results_data.get('gcn', {}).item() if 'gcn' in results_data else {}

    models_to_plot = {
        "PCA RFR": pca_results.get(target_name, {}).get('RFR', None),
        "PCA GBR": pca_results.get(target_name, {}).get('GBR', None),
        "Baseline RFR": baseline_results.get(target_name, {}).get('RFR', None),
        "Baseline GBR": baseline_results.get(target_name, {}).get('GBR', None),
        "GCN": gcn_results.get(target_name, {}).get('GCN', None)
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))  # Adjusted for 5 plots + 1 empty
    axes = axes.ravel()
    plot_idx = 0
    
    for model_name, data in models_to_plot.items():
        ax = axes[plot_idx]
        if data:
            true_vals = data['true_values']
            pred_vals = data['predictions']
            r2 = data.get('r2', np.nan)
            mse = data.get('mse', np.nan)
            
            ax.scatter(true_vals, pred_vals, alpha=0.5, s=15, label="Predictions")
            min_val = min(np.min(true_vals), np.min(pred_vals))
            max_val = max(np.max(true_vals), np.max(pred_vals))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal (y=x)")
            ax.set_xlabel("True " + target_name)
            ax.set_ylabel("Predicted " + target_name)
            ax.set_title(model_name + "\nR²: " + str(round(r2, 3)) + ", MSE: " + str(round(mse, 5)))
            ax.legend(loc='best')
            ax.grid(True, linestyle='--', alpha=0.7)
            plot_idx += 1
        else:
            ax.text(0.5, 0.5, "Data not available", ha='center', va='center', fontsize=12, color='red')
            ax.set_title(model_name)
            ax.set_xticks([])
            ax.set_yticks([])
            plot_idx += 1
            
    # Hide any unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')

    fig.suptitle("Predicted vs. True Values for " + target_name, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = os.path.join(DATABASE_PATH, "predicted_vs_true_" + target_name + "_" + str(plot_num_start) + "_" + TIMESTAMP + ".png")
    plt.savefig(filename, dpi=300)
    print("Saved plot: " + filename + " - Predicted vs. True for " + target_name + ".")
    plt.close(fig)
    return plot_num_start + 1


def plot_model_performance_comparison(results_data, plot_num_start=1):
    """Plots bar charts comparing R2 and MSE for all models."""
    print("Plotting model performance comparison (R2 and MSE)...")

    pca_results = results_data.get('pca_engineered', {}).item() if 'pca_engineered' in results_data else {}
    baseline_results = results_data.get('baseline_aggregated', {}).item() if 'baseline_aggregated' in results_data else {}
    gcn_results = results_data.get('gcn', {}).item() if 'gcn' in results_data else {}
    
    model_labels = ["PCA RFR", "PCA GBR", "Base RFR", "Base GBR", "GCN"]
    metrics = ['r2', 'mse']
    
    performance_data = {target: {metric: [] for metric in metrics} for target in TARGET_VARIABLES}

    for target in TARGET_VARIABLES:
        # PCA RFR
        rfr_pca_res = pca_results.get(target, {}).get('RFR', {})
        performance_data[target]['r2'].append(rfr_pca_res.get('r2', np.nan))
        performance_data[target]['mse'].append(rfr_pca_res.get('mse', np.nan))
        # PCA GBR
        gbr_pca_res = pca_results.get(target, {}).get('GBR', {})
        performance_data[target]['r2'].append(gbr_pca_res.get('r2', np.nan))
        performance_data[target]['mse'].append(gbr_pca_res.get('mse', np.nan))
        # Base RFR
        rfr_base_res = baseline_results.get(target, {}).get('RFR', {})
        performance_data[target]['r2'].append(rfr_base_res.get('r2', np.nan))
        performance_data[target]['mse'].append(rfr_base_res.get('mse', np.nan))
        # Base GBR
        gbr_base_res = baseline_results.get(target, {}).get('GBR', {})
        performance_data[target]['r2'].append(gbr_base_res.get('r2', np.nan))
        performance_data[target]['mse'].append(gbr_base_res.get('mse', np.nan))
        # GCN
        gcn_res_target = gcn_results.get(target, {}).get('GCN', {})
        performance_data[target]['r2'].append(gcn_res_target.get('r2', np.nan))
        performance_data[target]['mse'].append(gcn_res_target.get('mse', np.nan))

    # Plot R-squared
    fig_r2, axes_r2 = plt.subplots(1, 2, figsize=(18, 7), sharey=False)  # R2 can be negative
    for i, target in enumerate(TARGET_VARIABLES):
        ax = axes_r2[i]
        r2_values = performance_data[target]['r2']
        bars = ax.bar(model_labels, r2_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax.set_title("R² for " + target)
        ax.set_ylabel("R² Score")
        ax.set_xticklabels(model_labels, rotation=45, ha="right")
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Line at R2=0
        for bar in bars:  # Add text labels
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, str(round(yval,3)), va='bottom' if yval >=0 else 'top', ha='center')

    fig_r2.suptitle("Model Comparison: R-squared Scores", fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    filename_r2 = os.path.join(DATABASE_PATH, "model_performance_r2_" + str(plot_num_start) + "_" + TIMESTAMP + ".png")
    plt.savefig(filename_r2, dpi=300)
    print("Saved plot: " + filename_r2 + " - Comparison of R-squared scores for all models.")
    plt.close(fig_r2)

    # Plot MSE
    fig_mse, axes_mse = plt.subplots(1, 2, figsize=(18, 7), sharey=False)  # MSE varies
    for i, target in enumerate(TARGET_VARIABLES):
        ax = axes_mse[i]
        mse_values = performance_data[target]['mse']
        bars = ax.bar(model_labels, mse_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax.set_title("MSE for " + target)
        ax.set_ylabel("Mean Squared Error")
        ax.set_xticklabels(model_labels, rotation=45, ha="right")
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=0)  # MSE is non-negative
        for bar in bars:  # Add text labels
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, str(round(yval,5)), va='bottom', ha='center')


    fig_mse.suptitle("Model Comparison: Mean Squared Error", fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    filename_mse = os.path.join(DATABASE_PATH, "model_performance_mse_" + str(plot_num_start + 1) + "_" + TIMESTAMP + ".png")
    plt.savefig(filename_mse, dpi=300)
    print("Saved plot: " + filename_mse + " - Comparison of Mean Squared Error for all models.")
    plt.close(fig_mse)
    
    return plot_num_start + 2


# --- Main Execution ---
if __name__ == '__main__':
    current_plot_number = 1

    # 1. Distributions of engineered features
    engineered_features_train_path = os.path.join(DATABASE_PATH, 'engineered_features_train.npz')
    try:
        engineered_data_train = np.load(engineered_features_train_path, allow_pickle=True)
        X_train_engineered_raw = engineered_data_train['X']
        current_plot_number = plot_engineered_feature_distributions(X_train_engineered_raw, X_ENGINEERED_FEATURE_NAMES, current_plot_number)
    except FileNotFoundError:
        print("Warning: " + engineered_features_train_path + " not found. Skipping engineered feature distribution plots.")
    except Exception as e:
        print("Error plotting engineered feature distributions: " + str(e))


    # 2. PCA explained variance (already plotted in Step 2) and projection
    print("\nPCA explained variance plot was generated in Step 2 (pca_explained_variance_plot_1_<timestamp>.png).")
    pca_features_train_path = os.path.join(DATABASE_PATH, 'engineered_features_pca_train.npz')
    try:
        pca_data_train = np.load(pca_features_train_path, allow_pickle=True)
        X_train_pca = pca_data_train['X_pca']
        y_train_pca = pca_data_train['y']
        current_plot_number = plot_pca_projection(X_train_pca, y_train_pca, current_plot_number)
    except FileNotFoundError:
        print("Warning: " + pca_features_train_path + " not found. Skipping PCA projection plot.")
    except Exception as e:
        print("Error plotting PCA projection: " + str(e))

    # 3. Feature importances from tree-based models
    # Load models
    models_to_load = {
        "Omega_m": {
            "PCA_RFR": "RandomForest_PCA_Omega_m_model.pkl",
            "PCA_GBR": "GradientBoosting_PCA_Omega_m_model.pkl",
            "Baseline_RFR": "RandomForest_Baseline_Omega_m_model.pkl",
            "Baseline_GBR": "GradientBoosting_Baseline_Omega_m_model.pkl",
        },
        "sigma_8": {
            "PCA_RFR": "RandomForest_PCA_sigma_8_model.pkl",
            "PCA_GBR": "GradientBoosting_PCA_sigma_8_model.pkl",
            "Baseline_RFR": "RandomForest_Baseline_sigma_8_model.pkl",
            "Baseline_GBR": "GradientBoosting_Baseline_sigma_8_model.pkl",
        }
    }
    loaded_models = {target: {} for target in TARGET_VARIABLES}
    
    for target_name, model_files in models_to_load.items():
        for model_key, fname in model_files.items():
            try:
                loaded_models[target_name][model_key] = joblib.load(os.path.join(DATABASE_PATH, fname))
            except FileNotFoundError:
                print("Warning: Model file " + fname + " not found. Feature importances for this model will be skipped.")
                loaded_models[target_name][model_key] = None
            except Exception as e:
                print("Error loading model " + fname + ": " + str(e))
                loaded_models[target_name][model_key] = None


    num_pca_components = X_train_pca.shape[1] if 'X_train_pca' in locals() else 8  # Default if X_train_pca not loaded
    pca_feat_names = ["PC" + str(i+1) for i in range(num_pca_components)]

    for target_idx, target_name in enumerate(TARGET_VARIABLES):
        current_plot_number = plot_feature_importances(
            target_name,
            loaded_models[target_name].get("PCA_RFR"),
            loaded_models[target_name].get("PCA_GBR"),
            loaded_models[target_name].get("Baseline_RFR"),
            loaded_models[target_name].get("Baseline_GBR"),
            pca_feat_names,
            X_BASELINE_FEATURE_NAMES,
            current_plot_number
        )
        
    # 4. Predicted vs. true values for Omega_m and sigma_8 for all models
    all_results_path = os.path.join(DATABASE_PATH, 'all_model_results.npz')
    try:
        results_data = np.load(all_results_path, allow_pickle=True)
        for target_name in TARGET_VARIABLES:
            current_plot_number = plot_predicted_vs_true(target_name, results_data, current_plot_number)
        
        # 5. Comparison of model performance (bar plots of R², MSE for all approaches)
        current_plot_number = plot_model_performance_comparison(results_data, current_plot_number)

    except FileNotFoundError:
        print("Warning: " + all_results_path + " not found. Skipping predicted vs. true and performance comparison plots.")
    except Exception as e:
        print("Error processing results for plots: " + str(e))


    print("\nStep 4: Visualization and Plot Generation completed.")
    print("Total plots generated in this step (approx): " + str(current_plot_number -1))
