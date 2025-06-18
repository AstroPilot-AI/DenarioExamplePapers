# filename: codebase/step2_dimensionality_baseline_features.py
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import os
import time
import joblib  # For saving PCA model
from sklearn.model_selection import GroupShuffleSplit
import warnings  # MODIFIED: Import warnings module

# --- Configuration ---
DATABASE_PATH = "data/"
F_TREE_ORIGINAL = '/Users/fvillaescusa/Documents/Software/AstroPilot/data/Pablo_merger_trees2.pt'  # Path to original PyG dataset
PCA_N_COMPONENTS_VARIANCE = 0.95
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")

# Ensure DATABASE_PATH exists
os.makedirs(DATABASE_PATH, exist_ok=True)

# --- Load data from Step 1 ---
engineered_features_train_path = os.path.join(DATABASE_PATH, 'engineered_features_train.npz')
engineered_features_test_path = os.path.join(DATABASE_PATH, 'engineered_features_test.npz')
normalization_params_path = os.path.join(DATABASE_PATH, 'normalization_params.npz')

print("Loading engineered features and normalization parameters...")
try:
    train_data_engineered = np.load(engineered_features_train_path, allow_pickle=True)
    X_train_engineered = train_data_engineered['X']
    y_train = train_data_engineered['y']
    lh_ids_train = train_data_engineered['lh_ids']

    test_data_engineered = np.load(engineered_features_test_path, allow_pickle=True)
    X_test_engineered = test_data_engineered['X']
    y_test = test_data_engineered['y']
    lh_ids_test = test_data_engineered['lh_ids']

    normalization_params = np.load(normalization_params_path)
    node_feature_mean = normalization_params['mean']
    node_feature_std = normalization_params['std']
except FileNotFoundError as e:
    print("Error: Required data files from Step 1 not found. " + str(e))
    print("Please ensure Step 1 (data_preprocessing_feature_engineering.py) was executed successfully.")
    exit()

print("Shapes of loaded engineered features:")
print("  X_train_engineered: " + str(X_train_engineered.shape))
print("  y_train: " + str(y_train.shape))
print("  lh_ids_train: " + str(lh_ids_train.shape))
print("  X_test_engineered: " + str(X_test_engineered.shape))
print("  y_test: " + str(y_test.shape))
print("  lh_ids_test: " + str(lh_ids_test.shape))


# --- Handle NaNs in Engineered Features ---
print("Handling NaNs in engineered features using mean imputation...")
imputer = SimpleImputer(strategy='mean')
X_train_engineered_imputed = imputer.fit_transform(X_train_engineered)
X_test_engineered_imputed = imputer.transform(X_test_engineered)  # Use training set means for test set

num_nans_before_train = np.sum(np.isnan(X_train_engineered))
num_nans_after_train = np.sum(np.isnan(X_train_engineered_imputed))
num_nans_before_test = np.sum(np.isnan(X_test_engineered))
num_nans_after_test = np.sum(np.isnan(X_test_engineered_imputed))

print("NaNs in X_train_engineered before imputation: " + str(num_nans_before_train))
print("NaNs in X_train_engineered after imputation: " + str(num_nans_after_train))
print("NaNs in X_test_engineered before imputation: " + str(num_nans_before_test))
print("NaNs in X_test_engineered after imputation: " + str(num_nans_after_test))


# --- PCA on Engineered Features ---
print("Performing PCA on imputed engineered training features...")
# Fit PCA once to get all components for the plot
pca_full = PCA(random_state=42)
pca_full.fit(X_train_engineered_imputed)

# Fit PCA again with variance threshold
pca = PCA(n_components=PCA_N_COMPONENTS_VARIANCE, random_state=42)
X_train_pca = pca.fit_transform(X_train_engineered_imputed)
X_test_pca = pca.transform(X_test_engineered_imputed)

n_components_selected = pca.n_components_
print("Number of principal components selected to explain " + str(PCA_N_COMPONENTS_VARIANCE * 100) + "% variance: " + str(n_components_selected))
print("Shape of X_train_pca: " + str(X_train_pca.shape))
print("Shape of X_test_pca: " + str(X_test_pca.shape))
print("Total explained variance by selected components: " + str(np.sum(pca.explained_variance_ratio_)))

# Plot PCA Explained Variance
plt.figure(figsize=(10, 6))
plt.rcParams['text.usetex'] = False  # Ensure LaTeX is off
plt.plot(np.arange(1, pca_full.n_features_in_ + 1), np.cumsum(pca_full.explained_variance_ratio_), marker='o', linestyle='--', label='Cumulative Explained Variance')
plt.plot(np.arange(1, pca_full.n_features_in_ + 1), pca_full.explained_variance_ratio_, marker='x', linestyle='-', label='Individual Explained Variance')
plt.axhline(y=PCA_N_COMPONENTS_VARIANCE, color='r', linestyle=':', label=str(int(PCA_N_COMPONENTS_VARIANCE * 100)) + '% Variance Threshold')
plt.axvline(x=n_components_selected, color='g', linestyle=':', label=str(n_components_selected) + ' Components Selected')
plt.title('PCA Explained Variance for Engineered Features')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
pca_plot_filename = os.path.join(DATABASE_PATH, 'pca_explained_variance_plot_1_' + TIMESTAMP + '.png')
plt.savefig(pca_plot_filename, dpi=300)
print("PCA explained variance plot saved to " + pca_plot_filename)
plt.close()

# Save PCA model and transformed features
pca_model_path = os.path.join(DATABASE_PATH, 'pca_model.pkl')
joblib.dump(pca, pca_model_path)
print("PCA model saved to " + pca_model_path)

pca_train_features_path = os.path.join(DATABASE_PATH, 'engineered_features_pca_train.npz')
np.savez(pca_train_features_path, X_pca=X_train_pca, y=y_train, lh_ids=lh_ids_train)
print("PCA-transformed training features saved to " + pca_train_features_path)

pca_test_features_path = os.path.join(DATABASE_PATH, 'engineered_features_pca_test.npz')
np.savez(pca_test_features_path, X_pca=X_test_pca, y=y_test, lh_ids=lh_ids_test)
print("PCA-transformed testing features saved to " + pca_test_features_path)


# --- Construct Baseline Features ---
print("Constructing baseline features (aggregated node features)...")

print("Loading original full dataset for baseline feature construction...")
try:
    full_dataset_original = torch.load(F_TREE_ORIGINAL, map_location=torch.device('cpu'))
except Exception as e:
    print("Error loading original dataset: " + str(e))
    exit()

all_lh_ids_original = np.array([data.lh_id for data in full_dataset_original])
unique_lh_ids_original = np.unique(all_lh_ids_original)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_sim_idx, test_sim_idx = next(splitter.split(unique_lh_ids_original, groups=unique_lh_ids_original))
train_simulation_ids = unique_lh_ids_original[train_sim_idx]
test_simulation_ids = unique_lh_ids_original[test_sim_idx]

original_train_indices = [i for i, lh_id_val in enumerate(all_lh_ids_original) if lh_id_val in train_simulation_ids]
original_test_indices = [i for i, lh_id_val in enumerate(all_lh_ids_original) if lh_id_val in test_simulation_ids]

if X_train_engineered.shape[0] != len(original_train_indices):
    print("Warning: Number of rows in X_train_engineered (" + str(X_train_engineered.shape[0]) + ") " +
          "does not match length of original_train_indices (" + str(len(original_train_indices)) + "). " +
          "Baseline features might be misaligned if graphs were skipped in Step 1.")
if X_test_engineered.shape[0] != len(original_test_indices):
    print("Warning: Number of rows in X_test_engineered (" + str(X_test_engineered.shape[0]) + ") " +
          "does not match length of original_test_indices (" + str(len(original_test_indices)) + "). " +
          "Baseline features might be misaligned if graphs were skipped in Step 1.")


scaler_x_baseline = StandardScaler()
scaler_x_baseline.mean_ = node_feature_mean
scaler_x_baseline.scale_ = node_feature_std
if node_feature_std is not None and len(node_feature_std) > 0:
    scaler_x_baseline.var_ = node_feature_std**2
else:
    scaler_x_baseline.var_ = np.ones_like(node_feature_mean) if node_feature_mean is not None else np.array([])

scaler_x_baseline.n_features_in_ = len(node_feature_mean) if node_feature_mean is not None else 0


def get_baseline_features_for_graph(graph_data, scaler):
    """Computes aggregated node features for a single graph."""
    num_expected_node_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 4
    
    if not hasattr(graph_data, 'x') or graph_data.x is None or graph_data.x.shape[0] == 0:
        return np.full(num_expected_node_features * 4, np.nan)

    x_original = graph_data.x.numpy()
    if x_original.ndim == 1:
        x_original = x_original.reshape(1, -1)
    
    if x_original.shape[1] != num_expected_node_features:
        return np.full(num_expected_node_features * 4, np.nan)

    try:
        if not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_') or scaler.mean_ is None or scaler.scale_ is None:
            return np.full(num_expected_node_features * 4, np.nan)
        x_norm = scaler.transform(x_original)
    except Exception as e:
        return np.full(num_expected_node_features * 4, np.nan)

    if x_norm.shape[0] == 0:
        means = np.full(num_expected_node_features, np.nan)
        stds = np.full(num_expected_node_features, np.nan)
        mins = np.full(num_expected_node_features, np.nan)
        maxs = np.full(num_expected_node_features, np.nan)
    else:
        # MODIFIED: Use warnings module correctly
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered', RuntimeWarning)
            warnings.filterwarnings('ignore', r'Mean of empty slice', RuntimeWarning)
            warnings.filterwarnings('ignore', r'Degrees of freedom <= 0 for slice', RuntimeWarning)
            means = np.nanmean(x_norm, axis=0)
            stds = np.nanstd(x_norm, axis=0)
            mins = np.nanmin(x_norm, axis=0)
            maxs = np.nanmax(x_norm, axis=0)
            
            if x_norm.shape[0] == 1:
                stds = np.zeros_like(stds)

    return np.concatenate([means, stds, mins, maxs])

baseline_X_train_list = []
for original_idx in original_train_indices:
    graph = full_dataset_original[original_idx]
    baseline_feats = get_baseline_features_for_graph(graph, scaler_x_baseline)
    baseline_X_train_list.append(baseline_feats)
X_train_baseline = np.array(baseline_X_train_list)

baseline_X_test_list = []
for original_idx in original_test_indices:
    graph = full_dataset_original[original_idx]
    baseline_feats = get_baseline_features_for_graph(graph, scaler_x_baseline)
    baseline_X_test_list.append(baseline_feats)
X_test_baseline = np.array(baseline_X_test_list)

print("Shape of X_train_baseline: " + str(X_train_baseline.shape))
print("Shape of X_test_baseline: " + str(X_test_baseline.shape))

num_nans_baseline_train = np.sum(np.isnan(X_train_baseline))
num_nans_baseline_test = np.sum(np.isnan(X_test_baseline))
print("NaNs in X_train_baseline: " + str(num_nans_baseline_train))
if num_nans_baseline_train > 0:
    nan_rows_train_baseline = np.where(np.isnan(X_train_baseline).any(axis=1))[0]
    print("  Indices of rows with NaNs in X_train_baseline: " + str(nan_rows_train_baseline[:5]))
    nan_cols_train_baseline = np.where(np.isnan(X_train_baseline).any(axis=0))[0]
    print("  Indices of columns with NaNs in X_train_baseline: " + str(nan_cols_train_baseline))

print("NaNs in X_test_baseline: " + str(num_nans_baseline_test))
if num_nans_baseline_test > 0:
    nan_rows_test_baseline = np.where(np.isnan(X_test_baseline).any(axis=1))[0]
    print("  Indices of rows with NaNs in X_test_baseline: " + str(nan_rows_test_baseline[:5]))
    nan_cols_test_baseline = np.where(np.isnan(X_test_baseline).any(axis=0))[0]
    print("  Indices of columns with NaNs in X_test_baseline: " + str(nan_cols_test_baseline))


baseline_train_features_path = os.path.join(DATABASE_PATH, 'baseline_features_train.npz')
np.savez(baseline_train_features_path, X_baseline=X_train_baseline, y=y_train, lh_ids=lh_ids_train)
print("Baseline training features saved to " + baseline_train_features_path)

baseline_test_features_path = os.path.join(DATABASE_PATH, 'baseline_features_test.npz')
np.savez(baseline_test_features_path, X_baseline=X_test_baseline, y=y_test, lh_ids=lh_ids_test)
print("Baseline testing features saved to " + baseline_test_features_path)

print("Step 2: Dimensionality Reduction and Baseline Feature Construction completed.")