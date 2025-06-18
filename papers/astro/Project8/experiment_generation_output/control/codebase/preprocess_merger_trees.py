# filename: codebase/preprocess_merger_trees.py
import torch
from torch_geometric.data import Data
import os
import random
import numpy as np
from pathlib import Path


def preprocess_merger_trees():
    r"""
    Loads, preprocesses, and splits merger tree data.

    The preprocessing steps include:
    1. Loading raw data.
    2. Log-transforming mass and Vmax node features.
    3. Engineering edge features (mass ratio, time difference).
    4. Calculating an assembly bias proxy for each tree.
    5. Normalizing node and edge features across the dataset.
    6. Splitting data into training, validation, and test sets.
    7. Saving processed data and normalization parameters.
    """
    # Set seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # --- Configuration ---
    f_tree = '/Users/fvillaescusa/Documents/Software/AstroPilot/Project9/data/Pablo_merger_trees2.pt'
    output_dir = Path('data')
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path_train = output_dir / 'processed_merger_trees_train.pt'
    save_path_val = output_dir / 'processed_merger_trees_val.pt'
    save_path_test = output_dir / 'processed_merger_trees_test.pt'
    save_path_norm_params = output_dir / 'normalization_parameters.pt'

    # --- 1. Load Data ---
    print("Loading raw dataset from: " + str(f_tree))
    try:
        raw_dataset = torch.load(f_tree, weights_only=False)
    except Exception as e:
        print("Error loading dataset: " + str(e))
        return

    if not isinstance(raw_dataset, list) or not all(isinstance(item, Data) for item in raw_dataset):
        print("Loaded dataset is not in the expected format (list of torch_geometric.data.Data objects).")
        if isinstance(raw_dataset, Data):  # If it's a single Data object, wrap it in a list
             print("Loaded dataset is a single Data object. Wrapping it in a list.")
             raw_dataset = [raw_dataset]
        else:
            print("Type of loaded data: " + str(type(raw_dataset)))
            if raw_dataset and isinstance(raw_dataset, list):
                print("Type of first element: " + str(type(raw_dataset[0])))
            return


    print("Successfully loaded " + str(len(raw_dataset)) + " merger trees.")
    if not raw_dataset:
        print("Dataset is empty. Exiting.")
        return
        
    print("\n--- Example of raw data (first tree) ---")
    print(raw_dataset[0])
    # x: [num_nodes, 4] (mass, concentration, vmax, scale_factor)
    # edge_index: [2, num_edges]
    # edge_attr: [num_edges, 1] (original, will be replaced)
    # y: [1, 2] (Omega_m, sigma_8)
    # node_halo_id: [num_nodes, 1]
    # mask_main: [num_main_branch_nodes] (indices of nodes in main branch)

    # --- 2. Initial Processing and Feature Collection ---
    print("\n--- Starting Step 2: Initial Processing and Feature Collection ---")
    node_features_to_normalize_collector = [[] for _ in range(4)]  # log_mass, conc, log_vmax, sf
    edge_features_to_normalize_collector = [[] for _ in range(2)]  # mass_ratio, time_diff
    processed_data_storage = []
    assembly_bias_proxies_collected = []
    
    problematic_trees_nan_proxy = 0

    for tree_idx, raw_data in enumerate(raw_dataset):
        current_x = raw_data.x.clone().to(torch.float32)  # Ensure float32
        original_masses = current_x[:, 0].clone()      # Units: e.g., Msun/h
        original_scale_factors = current_x[:, 3].clone() # Dimensionless

        # Node Features - Log Transform (mass and Vmax)
        # Clamp to avoid log(0) or log(negative)
        current_x[:, 0] = torch.log10(torch.clamp(current_x[:, 0], min=1e-9))  # Log10(Mass)
        current_x[:, 2] = torch.log10(torch.clamp(current_x[:, 2], min=1e-9))  # Log10(Vmax)

        # Collect Node Features for Normalization
        node_features_to_normalize_collector[0].append(current_x[:, 0])  # Log10(Mass)
        node_features_to_normalize_collector[1].append(current_x[:, 1])  # Concentration (dimensionless)
        node_features_to_normalize_collector[2].append(current_x[:, 2])  # Log10(Vmax)
        node_features_to_normalize_collector[3].append(current_x[:, 3])  # Scale Factor (dimensionless)

        # Edge Feature Engineering (Raw)
        edge_index = raw_data.edge_index.to(torch.long)  # Ensure long
        source_nodes, target_nodes = edge_index[0], edge_index[1]
        
        masses_source = original_masses[source_nodes]
        masses_target = original_masses[target_nodes]
        
        min_masses = torch.minimum(masses_source, masses_target)
        max_masses = torch.maximum(masses_source, masses_target)
        
        mass_ratios = torch.zeros_like(min_masses)  # Initialize with zeros
        # Avoid division by zero: if max_masses is 0, ratio is 0. Otherwise, compute.
        valid_mask = max_masses > 1e-9  # Check against small epsilon
        mass_ratios[valid_mask] = min_masses[valid_mask] / max_masses[valid_mask]  # Dimensionless

        sf_source = original_scale_factors[source_nodes]
        sf_target = original_scale_factors[target_nodes]
        time_differences = torch.abs(sf_source - sf_target)  # Dimensionless

        current_edge_attr_raw = torch.stack([mass_ratios, time_differences], dim=1)

        # Collect Edge Features for Normalization
        if current_edge_attr_raw.numel() > 0:  # Only if there are edges
            edge_features_to_normalize_collector[0].append(mass_ratios)
            edge_features_to_normalize_collector[1].append(time_differences)
        
        # Assembly Bias Proxy Calculation
        assembly_bias_proxy = torch.tensor(float('nan'))  # Default to NaN
        if raw_data.mask_main is not None and len(raw_data.mask_main) > 0:
            main_branch_indices_np = raw_data.mask_main
            if isinstance(main_branch_indices_np, np.ndarray):
                 main_branch_indices = torch.from_numpy(main_branch_indices_np).squeeze()
            elif isinstance(main_branch_indices_np, torch.Tensor): 
                 main_branch_indices = main_branch_indices_np.squeeze()
            else: 
                 print("Warning: Tree " + str(tree_idx) + ": mask_main is of unexpected type: " + str(type(main_branch_indices_np)))
                 main_branch_indices = torch.empty(0, dtype=torch.long)

            if main_branch_indices.dim() == 0 and main_branch_indices.numel() == 1:  # Check if it's a 0-dim tensor with one element
                main_branch_indices = main_branch_indices.unsqueeze(0)  # Make it 1D tensor
            
            main_branch_indices = main_branch_indices.long()  # Ensure indices are long for indexing

            # Filter out-of-bounds indices from mask_main
            if main_branch_indices.numel() > 0:
                valid_indices_mask = main_branch_indices < raw_data.num_nodes
                main_branch_indices = main_branch_indices[valid_indices_mask]

            if main_branch_indices.numel() > 0:  # Check if mask_main is not empty after filtering
                main_branch_sfs = original_scale_factors[main_branch_indices]
                main_branch_masses = original_masses[main_branch_indices]
                
                # Select main halos at z=0 (sf >= 0.99)
                z0_mask = main_branch_sfs >= 0.99 
                z0_main_halos_masses = main_branch_masses[z0_mask]

                if len(z0_main_halos_masses) > 0:
                    assembly_bias_proxy = torch.mean(z0_main_halos_masses)
                else:
                    # Fallback: if no main halo at sf>=0.99, take the one with max sf in main branch
                    if len(main_branch_sfs) > 0:
                        max_sf_idx = torch.argmax(main_branch_sfs)
                        assembly_bias_proxy = main_branch_masses[max_sf_idx]
                    else: 
                        problematic_trees_nan_proxy += 1
            else: 
                problematic_trees_nan_proxy += 1
        else: 
            problematic_trees_nan_proxy += 1

        if not torch.isnan(assembly_bias_proxy):
            assembly_bias_proxies_collected.append(assembly_bias_proxy.item())
        
        processed_data_storage.append({
            'log_transformed_x': current_x,
            'edge_index': edge_index,
            'raw_edge_attr': current_edge_attr_raw,
            'assembly_bias_proxy': assembly_bias_proxy.reshape(1),  # Ensure [1] shape
            'num_nodes': raw_data.num_nodes,
            'lh_id': raw_data.lh_id if hasattr(raw_data, 'lh_id') else -1,
            'cosmo_params': raw_data.y.clone() if hasattr(raw_data, 'y') else torch.empty(0)
        })
    if problematic_trees_nan_proxy > 0:
        print("Warning: " + str(problematic_trees_nan_proxy) + " trees had issues with assembly bias proxy calculation (set to NaN initially, may have used fallback, or mask_main was problematic).")


    # --- 3. Calculate Normalization Parameters ---
    print("\n--- Starting Step 3: Calculating Normalization Parameters ---")
    node_norm_params = []
    node_feature_names = ["log10(Mass)", "Concentration", "log10(Vmax)", "Scale Factor"]
    for i, feature_list in enumerate(node_features_to_normalize_collector):
        if not feature_list: 
            print("Warning: Node feature '" + node_feature_names[i] + "' has no data for normalization.")
            node_norm_params.append({'min': torch.tensor(0.0), 'max': torch.tensor(1.0)}) 
            continue
        all_values = torch.cat(feature_list)
        min_val, max_val = torch.min(all_values), torch.max(all_values)
        node_norm_params.append({'min': min_val, 'max': max_val})
        print("Node Feature: " + node_feature_names[i])
        print("  Min before normalization: " + str(min_val.item()))
        print("  Max before normalization: " + str(max_val.item()))

    edge_norm_params = []
    edge_feature_names = ["Mass Ratio", "Time Difference"]
    for i, feature_list in enumerate(edge_features_to_normalize_collector):
        if not feature_list: 
            print("Warning: Edge feature '" + edge_feature_names[i] + "' has no data for normalization.")
            edge_norm_params.append({'min': torch.tensor(0.0), 'max': torch.tensor(1.0)}) 
            continue
        all_values = torch.cat(feature_list)
        min_val, max_val = torch.min(all_values), torch.max(all_values)
        edge_norm_params.append({'min': min_val, 'max': max_val})
        print("Edge Feature: " + edge_feature_names[i])
        print("  Min before normalization: " + str(min_val.item()))
        print("  Max before normalization: " + str(max_val.item()))

    # --- 4. Final Data Assembly (Normalization) ---
    print("\n--- Starting Step 4: Final Data Assembly (Normalization) ---")
    final_dataset_with_proxy = []
    skipped_trees_nan_proxy = 0

    for stored_item in processed_data_storage:
        if torch.isnan(stored_item['assembly_bias_proxy']).any():
            skipped_trees_nan_proxy += 1
            continue 

        normalized_x = stored_item['log_transformed_x'].clone()
        for i in range(4):  # Node features
            params = node_norm_params[i]
            min_v, max_v = params['min'], params['max']
            range_v = max_v - min_v
            if range_v > 1e-9: 
                normalized_x[:, i] = (normalized_x[:, i] - min_v) / range_v
            else:
                normalized_x[:, i] = torch.zeros_like(normalized_x[:, i])

        normalized_edge_attr = stored_item['raw_edge_attr'].clone()
        if normalized_edge_attr.numel() > 0: 
            for i in range(2):  # Edge features
                params = edge_norm_params[i]
                min_v, max_v = params['min'], params['max']
                range_v = max_v - min_v
                if range_v > 1e-9:
                    normalized_edge_attr[:, i] = (normalized_edge_attr[:, i] - min_v) / range_v
                else:
                    normalized_edge_attr[:, i] = torch.zeros_like(normalized_edge_attr[:, i])
        
        data_obj = Data(
            x=normalized_x,
            edge_index=stored_item['edge_index'],
            edge_attr=normalized_edge_attr,
            y=stored_item['assembly_bias_proxy'], 
            num_nodes=stored_item['num_nodes'],
            lh_id=torch.tensor([stored_item['lh_id']]), 
            cosmo_params=stored_item['cosmo_params'] 
        )
        final_dataset_with_proxy.append(data_obj)
    
    if skipped_trees_nan_proxy > 0:
        print(str(skipped_trees_nan_proxy) + " trees were skipped due to NaN assembly bias proxy.")
    
    if not final_dataset_with_proxy:
        print("No data remaining after filtering NaN proxies. Exiting.")
        return

    print("Number of trees in final processed dataset: " + str(len(final_dataset_with_proxy)))
    
    # --- Statistics of Assembly Bias Proxy ---
    if assembly_bias_proxies_collected: 
        proxies_tensor = torch.tensor(assembly_bias_proxies_collected, dtype=torch.float32)
        print("\n--- Assembly Bias Proxy Statistics (for trees where it's not NaN) ---")
        print("Mean: " + str(torch.mean(proxies_tensor).item()))
        print("Std: " + str(torch.std(proxies_tensor).item()))
        print("Min: " + str(torch.min(proxies_tensor).item()))
        print("Max: " + str(torch.max(proxies_tensor).item()))
    else:
        print("\n--- Assembly Bias Proxy Statistics ---")
        print("No valid assembly bias proxies were collected.")


    # --- 5. Data Splitting ---
    print("\n--- Starting Step 5: Data Splitting ---")
    num_total = len(final_dataset_with_proxy)
    indices = list(range(num_total))
    random.shuffle(indices)

    train_split = int(0.7 * num_total)
    val_split = int(0.15 * num_total)

    train_indices = indices[:train_split]
    val_indices = indices[train_split : train_split + val_split]
    test_indices = indices[train_split + val_split :]

    train_dataset = [final_dataset_with_proxy[i] for i in train_indices]
    val_dataset = [final_dataset_with_proxy[i] for i in val_indices]
    test_dataset = [final_dataset_with_proxy[i] for i in test_indices]

    print("Number of training samples: " + str(len(train_dataset)))
    print("Number of validation samples: " + str(len(val_dataset)))
    print("Number of test samples: " + str(len(test_dataset)))

    # --- 6. Save Data and Parameters ---
    print("\n--- Starting Step 6: Saving Data and Parameters ---")
    torch.save(train_dataset, save_path_train)
    print("Training dataset saved to: " + str(save_path_train))
    torch.save(val_dataset, save_path_val)
    print("Validation dataset saved to: " + str(save_path_val))
    torch.save(test_dataset, save_path_test)
    print("Test dataset saved to: " + str(save_path_test))

    normalization_params_to_save = {
        'node_norm_params': node_norm_params,
        'edge_norm_params': edge_norm_params
    }
    torch.save(normalization_params_to_save, save_path_norm_params)
    print("Normalization parameters saved to: " + str(save_path_norm_params))
    
    print("\n--- Example of processed data (first tree in training set if available) ---")
    if train_dataset:
        print(train_dataset[0])
        print("Processed x shape: " + str(train_dataset[0].x.shape))
        print("Processed edge_attr shape: " + str(train_dataset[0].edge_attr.shape))
        print("Processed y (assembly bias proxy): " + str(train_dataset[0].y))
    else:
        print("Training dataset is empty.")

    print("\nPreprocessing complete.")


if __name__ == '__main__':
    preprocess_merger_trees()