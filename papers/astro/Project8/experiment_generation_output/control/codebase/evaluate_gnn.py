# filename: codebase/evaluate_gnn.py
import torch
from torch_geometric.loader import DataLoader
from pathlib import Path
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from codebase.gcn_model_training import GCNNet  # Assuming GCNNet is in this file

def calculate_r_squared(y_true, y_pred):
    r"""
    Calculates the R-squared (coefficient of determination).

    Args:
        y_true (torch.Tensor): True target values.
        y_pred (torch.Tensor): Predicted values.

    Returns:
        float: R-squared value.
    """
    ss_res = torch.sum((y_true - y_pred)**2)
    ss_tot = torch.sum((y_true - torch.mean(y_true))**2)
    
    if ss_tot == 0:
        # Handle cases where y_true has no variance.
        # If y_pred also perfectly matches y_true (ss_res = 0), R2 is 1.
        # Otherwise (ss_res > 0), R2 is undefined or can be set to 0 or negative.
        # A common convention is 1.0 if ss_res is also 0, otherwise 0.0 or undefined.
        # For simplicity, if ss_tot is 0, and ss_res is also 0, R2 is 1. Otherwise, R2 is 0.
        return 1.0 if ss_res < 1e-9 else 0.0 
        
    r2 = 1 - (ss_res / ss_tot)
    return r2.item()

def evaluate_model_on_test_set(model, test_loader, criterion):
    r"""
    Evaluates the GNN model on the test set.

    Args:
        model (torch.nn.Module): The trained GNN model.
        test_loader (torch_geometric.loader.DataLoader): DataLoader for the test set.
        criterion (torch.nn.Module): The loss function (e.g., MSELoss).

    Returns:
        tuple: (all_predictions, all_targets, test_mse, test_r_squared)
    """
    model.eval()
    total_test_loss = 0
    num_test_batches = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_data in test_loader:
            out = model(batch_data)
            target = batch_data.y.unsqueeze(1).to(out.dtype)
            loss = criterion(out, target)
            
            if torch.isnan(loss):
                print("Warning: NaN loss encountered in test evaluation. Skipping batch contribution to MSE.")
            else:
                total_test_loss += loss.item() * batch_data.num_graphs  # Weighted by batch size
            
            num_test_batches += batch_data.num_graphs
            all_predictions.append(out.cpu())
            all_targets.append(target.cpu())

    if not all_predictions or not all_targets:
        print("Error: No predictions or targets collected from the test set.")
        return torch.empty(0), torch.empty(0), float('nan'), float('nan')

    all_predictions_tensor = torch.cat(all_predictions, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)

    test_mse = total_test_loss / num_test_batches if num_test_batches > 0 and not np.isnan(total_test_loss) else float('nan')
    test_r_squared = calculate_r_squared(all_targets_tensor, all_predictions_tensor)
    
    return all_predictions_tensor, all_targets_tensor, test_mse, test_r_squared

def plot_loss_curves(train_losses, val_losses, best_hyperparams_str, output_path):
    r"""
    Plots and saves the training and validation loss curves.

    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        best_hyperparams_str (str): String describing the best hyperparameters.
        output_path (Path): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error Loss')
    plt.title('Training and Validation Loss Curves\nBest Config: ' + best_hyperparams_str)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print("Saved training/validation loss curve plot to: " + str(output_path))
    print("Description: Plot shows the Mean Squared Error (MSE) loss on the training and validation sets " +
          "over epochs for the best hyperparameter configuration.")


def plot_predicted_vs_true(predictions, targets, mse, r_squared, output_path):
    r"""
    Plots and saves the predicted vs. true assembly bias values.

    Args:
        predictions (torch.Tensor): Model predictions on the test set.
        targets (torch.Tensor): True target values from the test set.
        mse (float): Mean Squared Error on the test set.
        r_squared (float): R-squared score on the test set.
        output_path (Path): Path to save the plot.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(targets.numpy(), predictions.numpy(), alpha=0.6, label='Predicted vs. True')
    
    min_val = min(targets.min().item(), predictions.min().item())
    max_val = max(targets.max().item(), predictions.max().item())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal (y=x)')
    
    plt.xlabel('True Assembly Bias Proxy')
    plt.ylabel('Predicted Assembly Bias Proxy')
    title_str = 'Predicted vs. True Assembly Bias on Test Set'
    if not np.isnan(mse) and not np.isnan(r_squared):
        title_str += '\nMSE: ' + str(round(mse, 4)) + ', R²: ' + str(round(r_squared, 4))
    plt.title(title_str)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Ensure aspect ratio is equal for y=x line to look correct
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print("Saved predicted vs. true assembly bias plot to: " + str(output_path))
    print("Description: Scatter plot of predicted assembly bias proxy against the true values for the test set. " +
          "The red dashed line represents a perfect prediction (y=x).")


if __name__ == '__main__':
    start_time_main = time.time()
    plt.rcParams['text.usetex'] = False  # Ensure LaTeX is not used

    # --- Configuration ---
    BASE_DIR = Path('.')
    DATA_DIR = BASE_DIR / 'data'
    MODEL_DIR = DATA_DIR / 'gnn_models'
    TUNING_LOG_PATH = DATA_DIR / 'gnn_tuning_results_log.pt'
    TEST_DATA_PATH = DATA_DIR / 'processed_merger_trees_test.pt'

    if not TUNING_LOG_PATH.exists():
        print("Error: Hyperparameter tuning log (" + str(TUNING_LOG_PATH) + ") not found. Run Step 3 first.")
        exit(1)
    if not TEST_DATA_PATH.exists():
        print("Error: Processed test data (" + str(TEST_DATA_PATH) + ") not found. Run Step 1 first.")
        exit(1)

    print("--- GNN Model Evaluation ---")

    # --- 1. Load Tuning Log and Identify Best Hyperparameters ---
    print("\nLoading hyperparameter tuning log...")
    try:
        tuning_results = torch.load(TUNING_LOG_PATH)
    except Exception as e:
        print("Error loading tuning log: " + str(e))
        exit(1)

    if not tuning_results:
        print("Error: Tuning log is empty.")
        exit(1)

    best_config_log = None
    overall_best_val_loss = float('inf')
    for config_log in tuning_results:
        # Ensure final_val_loss is a float, handle potential tensor
        current_val_loss = config_log['final_val_loss']
        if isinstance(current_val_loss, torch.Tensor):
            current_val_loss = current_val_loss.item()
            
        if not np.isnan(current_val_loss) and current_val_loss < overall_best_val_loss:
            overall_best_val_loss = current_val_loss
            best_config_log = config_log
            
    if best_config_log is None:
        print("Error: Could not determine the best hyperparameter configuration from the log (all runs might have failed or resulted in NaN/inf loss).")
        exit(1)

    best_hyperparams = {
        'num_gcn_layers': best_config_log['layers'],
        'hidden_channels': best_config_log['hidden']
    }
    best_hyperparams_str = "Layers: " + str(best_hyperparams['num_gcn_layers']) + ", HiddenCh: " + str(best_hyperparams['hidden_channels'])
    print("Best hyperparameters identified: " + best_hyperparams_str)
    print("Corresponding best validation loss: " + str(round(overall_best_val_loss, 4)))

    # --- 2. Load Test Data ---
    print("\nLoading test dataset...")
    try:
        test_dataset = torch.load(TEST_DATA_PATH)
    except Exception as e:
        print("Error loading test dataset: " + str(e))
        exit(1)

    if not test_dataset:
        print("Error: Test dataset is empty.")
        exit(1)
    print("Loaded " + str(len(test_dataset)) + " test samples.")

    # Determine feature dimensions from test data
    if test_dataset:
        first_data = test_dataset[0]
        num_node_features = first_data.num_node_features
        num_edge_features = 0 
        if hasattr(first_data, 'edge_attr') and first_data.edge_attr is not None and first_data.edge_attr.numel() > 0:
            if first_data.edge_attr.dim() > 1:
                num_edge_features = first_data.edge_attr.shape[1]
            elif first_data.edge_attr.dim() == 1:
                num_edge_features = 1
        print("Detected " + str(num_node_features) + " node features and " + str(num_edge_features) + " edge features from test data.")
    else:  # Should have exited if empty, but as a safeguard
        print("Error: Test dataset is empty, cannot determine feature dimensions.")
        exit(1)
        
    # Use a batch size that is reasonable for evaluation, e.g., 1 or more.
    # Given test set size is 5, batch_size can be 1 or 5.
    test_loader_batch_size = min(len(test_dataset), 4)  # Match training batch size or smaller
    if len(test_dataset) == 0:
        test_loader_batch_size = 1  # Avoid division by zero for empty dataset
        
    test_loader = DataLoader(test_dataset, batch_size=test_loader_batch_size, shuffle=False)


    # --- 3. Instantiate and Load Best Model ---
    print("\nInstantiating GCN model with best hyperparameters...")
    model = GCNNet(num_node_features=num_node_features,
                   num_edge_features=num_edge_features,
                   hidden_channels=best_hyperparams['hidden_channels'],
                   num_gcn_layers=best_hyperparams['num_gcn_layers'])

    model_weights_path = MODEL_DIR / ("layers_" + str(best_hyperparams['num_gcn_layers']) + "_hidden_" + str(best_hyperparams['hidden_channels'])) / 'best_model_weights.pt'
    
    if not model_weights_path.exists():
        print("Error: Best model weights file not found at " + str(model_weights_path))
        exit(1)
    
    try:
        model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
        print("Successfully loaded best model weights from: " + str(model_weights_path))
    except Exception as e:
        print("Error loading model weights: " + str(e))
        exit(1)

    # --- 4. Evaluate Model on Test Set ---
    print("\nEvaluating model on the test set...")
    criterion = torch.nn.MSELoss()
    predictions, targets, test_mse, test_r_squared = evaluate_model_on_test_set(model, test_loader, criterion)

    if np.isnan(test_mse) or np.isnan(test_r_squared):
        print("Test set evaluation resulted in NaN values. This might be due to issues during training or very small/problematic test data.")
    else:
        print("Test Set Performance:")
        print("  Mean Squared Error (MSE): " + str(round(test_mse, 4)))
        print("  R-squared (R²): " + str(round(test_r_squared, 4)))

    # --- 5. Generate Plots ---
    print("\nGenerating and saving plots...")
    timestamp = int(time.time())
    
    # Plot Training/Validation Loss Curves
    train_losses_best_config = best_config_log.get('train_losses', [])
    val_losses_best_config = best_config_log.get('val_losses', [])
    
    if not train_losses_best_config or not val_losses_best_config:
        print("Warning: Loss history not found in the log for the best configuration. Skipping loss curve plot.")
    else:
        loss_plot_filename = "training_validation_loss_curves_plot_1_" + str(timestamp) + ".png"
        loss_plot_path = DATA_DIR / loss_plot_filename
        plot_loss_curves(train_losses_best_config, val_losses_best_config, best_hyperparams_str, loss_plot_path)

    # Plot Predicted vs. True Assembly Bias
    if predictions.numel() > 0 and targets.numel() > 0:  # Check if predictions/targets are not empty
        pred_true_plot_filename = "predicted_vs_true_bias_plot_2_" + str(timestamp) + ".png"
        pred_true_plot_path = DATA_DIR / pred_true_plot_filename
        plot_predicted_vs_true(predictions, targets, test_mse, test_r_squared, pred_true_plot_path)
    else:
        print("Warning: No predictions/targets available to plot predicted vs. true values.")


    end_time_main = time.time()
    print("\nTotal time for GNN evaluation and plotting: " + str(round(end_time_main - start_time_main, 2)) + " seconds.")
    print("GNN model evaluation complete.")
