# filename: codebase/step3_regression_modeling.py
import numpy as np
import os
import time
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset  # Added Dataset for custom GCN dataset

# --- Configuration ---
DATABASE_PATH = "data/"
F_TREE_ORIGINAL = '/Users/fvillaescusa/Documents/Software/AstroPilot/data/Pablo_merger_trees2.pt'
RANDOM_STATE = 42
N_CV_SPLITS = 5  # Number of splits for GroupKFold
GCN_EPOCHS = 50 # Reduced epochs for faster example run, adjust as needed
GCN_BATCH_SIZE = 32
GCN_LEARNING_RATE = 1e-3

# Ensure DATABASE_PATH exists
os.makedirs(DATABASE_PATH, exist_ok=True)

# --- Helper Function for RFR/GBR Training and Evaluation ---
def train_evaluate_regressor(X_train, y_train_target, X_test, y_test_target,
                             groups_train, model, param_grid, model_name, target_name):
    """
    Trains and evaluates a regressor with hyperparameter tuning using GroupKFold.

    Args:
        X_train (np.ndarray): Training features.
        y_train_target (np.ndarray): Training target variable.
        X_test (np.ndarray): Testing features.
        y_test_target (np.ndarray): Testing target variable.
        groups_train (np.ndarray): Group labels for GroupKFold.
        model (sklearn estimator): The regressor instance.
        param_grid (dict): Hyperparameter grid for GridSearchCV.
        model_name (str): Name of the model (e.g., "RandomForest_PCA").
        target_name (str): Name of the target variable (e.g., "Omega_m").

    Returns:
        tuple: (y_pred_test, r2_test, mse_test, best_params)
    """
    print("\n--- Training " + model_name + " for " + target_name + " ---")
    
    # Handle potential NaNs in features if any slipped through (e.g. for baseline)
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        print("Warning: NaNs or Infs found in " + model_name + " X_train for " + target_name + ". Applying imputer.")
        imputer_train = SimpleImputer(strategy='mean')
        X_train = imputer_train.fit_transform(X_train)
        if np.isnan(X_test).any() or np.isinf(X_test).any():
             X_test = imputer_train.transform(X_test)  # Use imputer fitted on train

    group_kfold = GroupKFold(n_splits=N_CV_SPLITS)
    
    print("Performing GridSearchCV with GroupKFold (n_splits=" + str(N_CV_SPLITS) + ")...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=group_kfold, scoring='r2', n_jobs=-1, verbose=0)  # verbose=0 to reduce output
    grid_search.fit(X_train, y_train_target, groups=groups_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Best hyperparameters: " + str(best_params))

    y_pred_test = best_model.predict(X_test)
    r2_test = r2_score(y_test_target, y_pred_test)
    mse_test = mean_squared_error(y_test_target, y_pred_test)

    print(model_name + " for " + target_name + " - Test R²: " + str(r2_test))
    print(model_name + " for " + target_name + " - Test MSE: " + str(mse_test))

    # Save the trained model
    model_filename = os.path.join(DATABASE_PATH, model_name + "_" + target_name + "_model.pkl")
    joblib.dump(best_model, model_filename)
    print("Saved trained model to " + model_filename)

    return y_pred_test, r2_test, mse_test, best_params

# --- Load Data from Previous Steps ---
print("Loading data from previous steps...")

# PCA-transformed engineered features
pca_train_path = os.path.join(DATABASE_PATH, 'engineered_features_pca_train.npz')
pca_test_path = os.path.join(DATABASE_PATH, 'engineered_features_pca_test.npz')
train_pca_data = np.load(pca_train_path)
X_train_pca = train_pca_data['X_pca']
y_train_pca = train_pca_data['y']
lh_ids_train_pca = train_pca_data['lh_ids']
test_pca_data = np.load(pca_test_path)
X_test_pca = test_pca_data['X_pca']
y_test_pca = test_pca_data['y']
# lh_ids_test_pca = test_pca_data['lh_ids'] # Not strictly needed for evaluation if model is already trained

# Baseline aggregated node features
baseline_train_path = os.path.join(DATABASE_PATH, 'baseline_features_train.npz')
baseline_test_path = os.path.join(DATABASE_PATH, 'baseline_features_test.npz')
train_baseline_data = np.load(baseline_train_path)
X_train_baseline = train_baseline_data['X_baseline']
y_train_baseline = train_baseline_data['y']
lh_ids_train_baseline = train_baseline_data['lh_ids']
test_baseline_data = np.load(baseline_test_path)
X_test_baseline = test_baseline_data['X_baseline']
y_test_baseline = test_baseline_data['y']
# lh_ids_test_baseline = test_baseline_data['lh_ids']

# Target names
target_variables = ['Omega_m', 'sigma_8']
all_results = {}

# --- Define Hyperparameter Grids ---
# Smaller grids for faster execution, expand for thorough search
rfr_param_grid = {
    'n_estimators': [50, 100], # Reduced from [100, 200]
    'max_depth': [None, 10],    # Reduced from [10, 20]
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 3]
}

gbr_param_grid = {
    'n_estimators': [50, 100], # Reduced from [100, 200]
    'learning_rate': [0.05, 0.1], # Reduced from [0.01, 0.1]
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}


# --- 1. Modeling with PCA-Reduced Engineered Features ---
print("\n=== Modeling with PCA-Reduced Engineered Features ===")
all_results['pca_engineered'] = {}
for i, target_name in enumerate(target_variables):
    all_results['pca_engineered'][target_name] = {}
    
    # Random Forest
    rfr_pca = RandomForestRegressor(random_state=RANDOM_STATE)
    y_pred_rfr_pca, r2_rfr_pca, mse_rfr_pca, params_rfr_pca = train_evaluate_regressor(
        X_train_pca, y_train_pca[:, i], X_test_pca, y_test_pca[:, i],
        lh_ids_train_pca, rfr_pca, rfr_param_grid,
        "RandomForest_PCA", target_name
    )
    all_results['pca_engineered'][target_name]['RFR'] = {
        'predictions': y_pred_rfr_pca, 'true_values': y_test_pca[:, i],
        'r2': r2_rfr_pca, 'mse': mse_rfr_pca, 'best_params': params_rfr_pca
    }

    # Gradient Boosting
    gbr_pca = GradientBoostingRegressor(random_state=RANDOM_STATE)
    y_pred_gbr_pca, r2_gbr_pca, mse_gbr_pca, params_gbr_pca = train_evaluate_regressor(
        X_train_pca, y_train_pca[:, i], X_test_pca, y_test_pca[:, i],
        lh_ids_train_pca, gbr_pca, gbr_param_grid,
        "GradientBoosting_PCA", target_name
    )
    all_results['pca_engineered'][target_name]['GBR'] = {
        'predictions': y_pred_gbr_pca, 'true_values': y_test_pca[:, i],
        'r2': r2_gbr_pca, 'mse': mse_gbr_pca, 'best_params': params_gbr_pca
    }

# --- 2. Modeling with Baseline Aggregated Node Features ---
print("\n=== Modeling with Baseline Aggregated Node Features ===")
all_results['baseline_aggregated'] = {}
for i, target_name in enumerate(target_variables):
    all_results['baseline_aggregated'][target_name] = {}

    # Random Forest
    rfr_baseline = RandomForestRegressor(random_state=RANDOM_STATE)
    y_pred_rfr_base, r2_rfr_base, mse_rfr_base, params_rfr_base = train_evaluate_regressor(
        X_train_baseline, y_train_baseline[:, i], X_test_baseline, y_test_baseline[:, i],
        lh_ids_train_baseline, rfr_baseline, rfr_param_grid,
        "RandomForest_Baseline", target_name
    )
    all_results['baseline_aggregated'][target_name]['RFR'] = {
        'predictions': y_pred_rfr_base, 'true_values': y_test_baseline[:, i],
        'r2': r2_rfr_base, 'mse': mse_rfr_base, 'best_params': params_rfr_base
    }

    # Gradient Boosting
    gbr_baseline = GradientBoostingRegressor(random_state=RANDOM_STATE)
    y_pred_gbr_base, r2_gbr_base, mse_gbr_base, params_gbr_base = train_evaluate_regressor(
        X_train_baseline, y_train_baseline[:, i], X_test_baseline, y_test_baseline[:, i],
        lh_ids_train_baseline, gbr_baseline, gbr_param_grid,
        "GradientBoosting_Baseline", target_name
    )
    all_results['baseline_aggregated'][target_name]['GBR'] = {
        'predictions': y_pred_gbr_base, 'true_values': y_test_baseline[:, i],
        'r2': r2_gbr_base, 'mse': mse_gbr_base, 'best_params': params_gbr_base
    }


# --- 3. GCN Baseline (Optional) ---
print("\n=== GCN Baseline Model ===")
all_results['gcn'] = {target_variables[0]: {}, target_variables[1]: {}}

# Load original dataset
try:
    print("Loading original PyG dataset for GCN...")
    full_original_dataset = torch.load(F_TREE_ORIGINAL, map_location=torch.device('cpu'))
except Exception as e:
    print("Error loading original dataset for GCN: " + str(e))
    full_original_dataset = []  # Ensure it's an iterable

if not isinstance(full_original_dataset, list) or not all(isinstance(g, Data) for g in full_original_dataset if g is not None):
    print("Warning: Original dataset is not in the expected PyG list format. Skipping GCN.")
    full_original_dataset = []


if len(full_original_dataset) > 0:
    # MODIFIED: Handle lh_id being int or tensor
    all_lh_ids_original = np.array([
        data.lh_id if isinstance(data.lh_id, int) else data.lh_id.item()
        for data in full_original_dataset
        if hasattr(data, 'lh_id') and data.lh_id is not None
    ])
    
    if len(all_lh_ids_original) != len(full_original_dataset):
        print("Warning: Some graphs in original dataset are missing 'lh_id' or have None 'lh_id'. Filtering them out for GCN.")
        valid_indices_for_gcn = [
            i for i, data in enumerate(full_original_dataset)
            if hasattr(data, 'lh_id') and data.lh_id is not None
        ]
        full_original_dataset = [full_original_dataset[i] for i in valid_indices_for_gcn]
        # Recompute all_lh_ids_original after filtering
        all_lh_ids_original = np.array([
            data.lh_id if isinstance(data.lh_id, int) else data.lh_id.item()
            for data in full_original_dataset
        ])


    unique_lh_ids_original = np.unique(all_lh_ids_original)

    if len(unique_lh_ids_original) > 0:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        train_sim_idx, test_sim_idx = next(gss.split(unique_lh_ids_original, groups=unique_lh_ids_original))
        
        train_simulation_ids = unique_lh_ids_original[train_sim_idx]
        test_simulation_ids = unique_lh_ids_original[test_sim_idx]

        gcn_train_indices = [i for i, lh_id_val in enumerate(all_lh_ids_original) if lh_id_val in train_simulation_ids]
        gcn_test_indices = [i for i, lh_id_val in enumerate(all_lh_ids_original) if lh_id_val in test_simulation_ids]

        gcn_train_graphs = [full_original_dataset[i] for i in gcn_train_indices]
        gcn_test_graphs = [full_original_dataset[i] for i in gcn_test_indices]
        
        print("GCN: Number of training graphs: " + str(len(gcn_train_graphs)))
        print("GCN: Number of testing graphs: " + str(len(gcn_test_graphs)))

        # Load normalization parameters
        norm_params_path = os.path.join(DATABASE_PATH, 'normalization_params.npz')
        norm_params = np.load(norm_params_path)
        node_feat_mean = torch.tensor(norm_params['mean'], dtype=torch.float32)
        node_feat_std = torch.tensor(norm_params['std'], dtype=torch.float32)
        
        # Custom Dataset for GCN to handle normalization and ensure 'y' is float
        class NormalizedGraphDataset(Dataset):
            def __init__(self, graph_list, node_mean, node_std):
                super(NormalizedGraphDataset, self).__init__()
                self.graph_list = []
                for g in graph_list:
                    if not hasattr(g, 'x') or g.x is None or g.x.shape[0] == 0:
                        # print("Skipping graph with no node features for GCN.")
                        continue
                    if not hasattr(g, 'y') or g.y is None:
                        # print("Skipping graph with no y attribute for GCN.")
                        continue
                    
                    new_g = g.clone()  # Clone to avoid modifying original data
                    new_g.x = (new_g.x - node_mean) / (node_std + 1e-7)  # Normalize node features
                    new_g.y = new_g.y.float()  # Ensure y is float
                    self.graph_list.append(new_g)
            
            def len(self):
                return len(self.graph_list)
            
            def get(self, idx):
                return self.graph_list[idx]
            
            @property  # Added this property
            def num_node_features(self):
                if len(self.graph_list) == 0:
                    return 0  # Or raise an error, or return a default
                return self.graph_list[0].num_node_features


        if len(gcn_train_graphs) > 0 and len(gcn_test_graphs) > 0:
            gcn_train_dataset = NormalizedGraphDataset(gcn_train_graphs, node_feat_mean, node_feat_std)
            gcn_test_dataset = NormalizedGraphDataset(gcn_test_graphs, node_feat_mean, node_feat_std)

            gcn_train_loader = DataLoader(gcn_train_dataset, batch_size=GCN_BATCH_SIZE, shuffle=True)
            gcn_test_loader = DataLoader(gcn_test_dataset, batch_size=GCN_BATCH_SIZE, shuffle=False)

            # GCN Model Definition
            class GCN(torch.nn.Module):
                def __init__(self, num_node_features, hidden_channels, num_classes):
                    super(GCN, self).__init__()
                    torch.manual_seed(RANDOM_STATE)
                    self.conv1 = GCNConv(num_node_features, hidden_channels)
                    self.conv2 = GCNConv(hidden_channels, hidden_channels)
                    self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
                    self.lin2 = torch.nn.Linear(hidden_channels // 2, num_classes)

                def forward(self, x, edge_index, batch):
                    x = self.conv1(x, edge_index)
                    x = F.relu(x)
                    x = self.conv2(x, edge_index)
                    x = F.relu(x)
                    x = global_mean_pool(x, batch)  # Graph-level embedding
                    x = F.relu(self.lin1(x))
                    x = self.lin2(x)  # No activation for regression output
                    return x

            num_node_features = gcn_train_dataset.num_node_features if len(gcn_train_dataset) > 0 else 4  # Fallback
            gcn_model = GCN(num_node_features=num_node_features, hidden_channels=64, num_classes=2)
            optimizer = torch.optim.Adam(gcn_model.parameters(), lr=GCN_LEARNING_RATE)
            criterion = torch.nn.MSELoss()

            print("Training GCN model...")
            gcn_model.train()
            for epoch in range(GCN_EPOCHS):
                total_loss = 0
                for data_batch in gcn_train_loader:
                    optimizer.zero_grad()
                    out = gcn_model(data_batch.x, data_batch.edge_index, data_batch.batch)
                    loss = criterion(out, data_batch.y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * data_batch.num_graphs
                # if (epoch + 1) % 10 == 0: # Reduced print frequency
                #     print("GCN Epoch " + str(epoch+1) + "/" + str(GCN_EPOCHS) + ", Avg Loss: " + str(total_loss / len(gcn_train_loader.dataset)))
            print("GCN training finished.")

            # Evaluate GCN
            gcn_model.eval()
            all_gcn_preds = []
            all_gcn_true = []
            with torch.no_grad():
                for data_batch in gcn_test_loader:
                    out = gcn_model(data_batch.x, data_batch.edge_index, data_batch.batch)
                    all_gcn_preds.append(out.cpu().numpy())
                    all_gcn_true.append(data_batch.y.cpu().numpy())
            
            all_gcn_preds = np.concatenate(all_gcn_preds, axis=0)
            all_gcn_true = np.concatenate(all_gcn_true, axis=0)

            for i, target_name in enumerate(target_variables):
                r2_gcn = r2_score(all_gcn_true[:, i], all_gcn_preds[:, i])
                mse_gcn = mean_squared_error(all_gcn_true[:, i], all_gcn_preds[:, i])
                print("GCN for " + target_name + " - Test R²: " + str(r2_gcn))
                print("GCN for " + target_name + " - Test MSE: " + str(mse_gcn))
                all_results['gcn'][target_name]['GCN'] = {
                    'predictions': all_gcn_preds[:, i], 'true_values': all_gcn_true[:, i],
                    'r2': r2_gcn, 'mse': mse_gcn
                }
            
            gcn_model_path = os.path.join(DATABASE_PATH, "gcn_model.pt")
            torch.save(gcn_model.state_dict(), gcn_model_path)
            print("Saved GCN model to " + gcn_model_path)
        else:
            print("Skipping GCN training as no valid training/testing graphs found after filtering.")
    else:
        print("Skipping GCN as no unique simulation IDs found in the original dataset.")
else:
    print("Skipping GCN as original dataset could not be loaded or is empty.")


# --- Save All Results ---
results_path = os.path.join(DATABASE_PATH, 'all_model_results.npz')
# Convert params to string for saving in npz, as it doesn't handle dicts well
for model_type_key in all_results:
    for target_key in all_results[model_type_key]:
        for model_key in all_results[model_type_key][target_key]:
            if 'best_params' in all_results[model_type_key][target_key][model_key]:
                all_results[model_type_key][target_key][model_key]['best_params'] = \
                    str(all_results[model_type_key][target_key][model_key]['best_params'])

np.savez_compressed(results_path, **all_results)  # Using savez_compressed
print("\nAll model results (predictions, true values, metrics) saved to " + results_path)

print("\n--- Summary of Performance ---")
for model_type, target_data in all_results.items():
    print("\nFeature Set: " + model_type)
    for target_name, models in target_data.items():
        print("  Target: " + target_name)
        for model_name, metrics in models.items():
            print("    Model: " + model_name + " - R²: " + str(round(metrics['r2'], 4)) + ", MSE: " + str(round(metrics['mse'], 6)))

print("\nStep 3: Regression Modeling and Evaluation completed.")