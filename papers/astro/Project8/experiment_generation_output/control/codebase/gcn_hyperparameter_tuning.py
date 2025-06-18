# filename: codebase/gcn_hyperparameter_tuning.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from pathlib import Path
import time
import os
import numpy as np


class GCNNet(torch.nn.Module):
    r"""
    Graph Convolutional Network (GCN) for predicting a scalar value from graph data.

    The network consists of:
    1. Encoders for node and edge features.
    2. A configurable number of GCN layers.
    3. A global mean pooling layer.
    4. A final linear regression layer.
    """
    def __init__(self, num_node_features, num_edge_features, hidden_channels, num_gcn_layers):
        r"""
        Initializes the GCNNet model.

        Args:
            num_node_features (int): Dimensionality of input node features.
            num_edge_features (int): Dimensionality of input edge features.
            hidden_channels (int): Number of hidden units in GCN layers and encoders.
            num_gcn_layers (int): Number of GCN layers.
        """
        super(GCNNet, self).__init__()
        torch.manual_seed(42)  # For reproducibility of weight initialization

        self.node_encoder = torch.nn.Linear(num_node_features, hidden_channels)
        if num_edge_features > 0:
            self.edge_encoder = torch.nn.Linear(num_edge_features, hidden_channels)
        else:
            self.edge_encoder = None  # No edge features to encode

        self.gcn_layers = torch.nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))
        
        self.regressor = torch.nn.Linear(hidden_channels, 1)  # Predict a single scalar

    def forward(self, data):
        r"""
        Forward pass of the GCNNet.

        Args:
            data (torch_geometric.data.Data): Input graph data object.
                Must contain:
                - x: Node features [num_nodes, num_node_features]
                - edge_index: Graph connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, num_edge_features] (optional)
                - batch: Batch vector [num_nodes] (for batching graphs)

        Returns:
            torch.Tensor: Predicted scalar value for each graph in the batch. [batch_size, 1]
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.node_encoder(x)
        
        encoded_edge_attr = None
        if self.edge_encoder and edge_attr is not None and edge_attr.numel() > 0:
            encoded_edge_attr = self.edge_encoder(edge_attr)

        for gcn_layer in self.gcn_layers:
            current_edge_weight = None
            if encoded_edge_attr is not None:
                if encoded_edge_attr.dim() > 1 and encoded_edge_attr.shape[1] > 1:
                    # Simple aggregation if edge features are multi-dimensional
                    current_edge_weight = encoded_edge_attr.mean(dim=1)
                else:
                    current_edge_weight = encoded_edge_attr.squeeze()

            x = gcn_layer(x, edge_index, edge_weight=current_edge_weight)
            x = F.relu(x)
            # Dropout could be added here: x = F.dropout(x, p=0.5, training=self.training)

        x = global_mean_pool(x, batch)  # Aggregate node embeddings to graph-level
        x = self.regressor(x)
        return x


def train_gnn_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, model_save_dir):
    r"""
    Trains and validates the GNN model.

    Args:
        model (torch.nn.Module): The GNN model to train.
        train_loader (torch_geometric.loader.DataLoader): DataLoader for the training set.
        val_loader (torch_geometric.loader.DataLoader): DataLoader for the validation set.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (torch.nn.Module): The loss function.
        num_epochs (int): Number of epochs to train for.
        model_save_dir (Path): Directory to save the best model weights.

    Returns:
        tuple: (list of average training losses per epoch, list of average validation losses per epoch)
    """
    model_save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print("\n--- Starting Model Training ---")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        for batch_data in train_loader:
            optimizer.zero_grad()
            out = model(batch_data)
            loss = criterion(out, batch_data.y.unsqueeze(1))  # Ensure target is [batch_size, 1]
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            num_train_batches += 1
        
        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        with torch.no_grad():
            for batch_data in val_loader:
                out = model(batch_data)
                loss = criterion(out, batch_data.y.unsqueeze(1))
                total_val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0
        val_losses.append(avg_val_loss)

        print("Epoch " + str(epoch + 1) + "/" + str(num_epochs) + ": Train Loss: " + str(round(avg_train_loss, 4)) + ", Val Loss: " + str(round(avg_val_loss, 4)))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = model_save_dir / 'best_model_weights.pt'
            torch.save(model.state_dict(), best_model_path)
            print("  Saved new best model to " + str(best_model_path))
            
    print("--- Model Training Complete ---")
    print("Best validation loss achieved: " + str(round(best_val_loss, 4)))
    return train_losses, val_losses


if __name__ == '__main__':
    start_time_main = time.time()

    # --- Configuration ---
    BASE_DIR = Path('.')
    DATA_DIR = BASE_DIR / 'data'
    MODEL_SAVE_DIR = BASE_DIR / 'data' / 'gnn_models'  # For saving model weights
    
    # Ensure data directory exists from previous steps
    if not DATA_DIR.exists():
        print("Error: Data directory " + str(DATA_DIR) + " not found. Please run preprocessing first.")
        exit(1)

    TRAIN_DATA_PATH = DATA_DIR / 'processed_merger_trees_train.pt'
    VAL_DATA_PATH = DATA_DIR / 'processed_merger_trees_val.pt'

    # Hyperparameters for the GNN and training
    LEARNING_RATE = 0.001  # Tunable
    BATCH_SIZE = 4         # Tunable (small due to small dataset size)
    NUM_EPOCHS = 50        # Tunable (increased for potentially better convergence)
    WEIGHT_DECAY = 1e-5    # Tunable

    # Hyperparameter grid for tuning
    param_grid = {
        'num_gcn_layers': [1, 2],       # Tunable
        'hidden_channels': [16, 32]     # Tunable
    }
    
    print("--- GNN Model Design and Training Setup ---")

    # --- 1. Load Data ---
    print("\nLoading preprocessed datasets...")
    if not TRAIN_DATA_PATH.exists() or not VAL_DATA_PATH.exists():
        print("Error: Processed train/validation data not found. Run preprocessing (Step 1) and TDA (Step 2).")
        exit(1)
    
    try:
        train_dataset = torch.load(TRAIN_DATA_PATH)
        val_dataset = torch.load(VAL_DATA_PATH)
    except Exception as e:
        print("Error loading datasets: " + str(e))
        exit(1)

    if not train_dataset or not val_dataset:
        print("Error: Loaded datasets are empty.")
        exit(1)
        
    print("Loaded " + str(len(train_dataset)) + " training samples and " + str(len(val_dataset)) + " validation samples.")

    # Determine feature dimensions from the first data object
    # Assuming all data objects have the same feature dimensions
    if train_dataset:
        first_data = train_dataset[0]
        num_node_features = first_data.num_node_features
        num_edge_features = first_data.num_edge_features if hasattr(first_data, 'num_edge_features') and first_data.edge_attr is not None and first_data.edge_attr.numel() > 0 else 0
        if first_data.edge_attr is not None and first_data.edge_attr.dim() > 1:
            num_edge_features = first_data.edge_attr.shape[1]
        elif first_data.edge_attr is not None and first_data.edge_attr.dim() == 1 and first_data.edge_attr.numel() > 0:
            num_edge_features = 1  # Treat as a single edge feature
        else:
            num_edge_features = 0

        print("Detected " + str(num_node_features) + " node features.")
        print("Detected " + str(num_edge_features) + " edge features.")
    else:
        print("Error: Training dataset is empty, cannot determine feature dimensions.")
        exit(1)

    # --- 2. Create DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 3. Hyperparameter Tuning Loop ---
    print("\n--- Starting Hyperparameter Tuning ---")
    best_hyperparams = None
    overall_best_val_loss = float('inf')
    
    results_log = []

    for n_layers in param_grid['num_gcn_layers']:
        for h_channels in param_grid['hidden_channels']:
            current_hyperparams_str = "Layers: " + str(n_layers) + ", HiddenCh: " + str(h_channels)
            print("\nTraining with: " + current_hyperparams_str)
            
            current_model_save_dir = MODEL_SAVE_DIR / ("layers_" + str(n_layers) + "_hidden_" + str(h_channels))

            # Initialize model
            model = GCNNet(num_node_features=num_node_features,
                           num_edge_features=num_edge_features,
                           hidden_channels=h_channels,
                           num_gcn_layers=n_layers)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            criterion = torch.nn.MSELoss()

            # Train and validate
            train_losses, val_losses = train_gnn_model(model, train_loader, val_loader, 
                                                       optimizer, criterion, NUM_EPOCHS, 
                                                       current_model_save_dir)
            
            final_val_loss_for_config = val_losses[-1] if val_losses else float('inf')
            print("Final validation loss for " + current_hyperparams_str + ": " + str(round(final_val_loss_for_config, 4)))
            results_log.append({
                'layers': n_layers, 'hidden': h_channels, 
                'final_val_loss': final_val_loss_for_config,
                'train_losses': train_losses, 'val_losses': val_losses
            })

            if final_val_loss_for_config < overall_best_val_loss:
                overall_best_val_loss = final_val_loss_for_config
                best_hyperparams = {'num_gcn_layers': n_layers, 'hidden_channels': h_channels, 'lr': LEARNING_RATE, 'batch_size': BATCH_SIZE}
                print("!!! New best validation loss found with these hyperparameters !!!")

    print("\n--- Hyperparameter Tuning Complete ---")
    if best_hyperparams:
        print("Best hyperparameters found:")
        print("  Number of GCN Layers: " + str(best_hyperparams['num_gcn_layers']))
        print("  Hidden Channels: " + str(best_hyperparams['hidden_channels']))
        print("  Learning Rate: " + str(best_hyperparams['lr']))
        print("  Batch Size: " + str(best_hyperparams['batch_size']))
        print("  Corresponding best validation loss: " + str(round(overall_best_val_loss, 4)))
    else:
        print("Hyperparameter tuning did not yield a best configuration (e.g., all runs failed or produced NaN/inf loss).")

    # Save results log (e.g. to a file for later analysis of loss curves)
    results_log_path = DATA_DIR / 'gnn_tuning_results_log.pt'
    torch.save(results_log, results_log_path)
    print("Saved hyperparameter tuning log to: " + str(results_log_path))

    end_time_main = time.time()
    print("\nTotal time for GNN design, training setup, and hyperparameter tuning: " + str(round(end_time_main - start_time_main, 2)) + " seconds.")
    print("GNN model design and training pipeline setup complete. Model weights for different hyperparameter configurations (best epoch per config) are saved in " + str(MODEL_SAVE_DIR))
