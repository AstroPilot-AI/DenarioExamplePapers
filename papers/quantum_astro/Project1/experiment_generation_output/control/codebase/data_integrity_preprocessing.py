# filename: codebase/data_integrity_preprocessing.py
import os
import torch
# numpy is not strictly needed here as torch can handle numerical operations and string conversions.

# Configuration
f_tree = '/Users/fvillaescusa/Documents/Software/AstroPilot/Project6/data/Pablo_merger_trees.pt'
output_dir = 'data'
preprocessed_data_path = os.path.join(output_dir, 'preprocessed_merger_trees.pt')
scaling_params_path = os.path.join(output_dir, 'feature_scaling_params.pt')
log_epsilon = 1e-7 # Small value to clamp to before log transform for non-positive values.
std_epsilon = 1e-9   # Small value to avoid division by zero if std is too small.

# Feature names and indices for clarity
feature_names = ['mass', 'concentration', 'vmax', 'scale_factor']
log_transform_indices = [0, 2] # mass and vmax

def get_feature_stats(all_features_tensor, feature_names_list):
    r"""Computes and prints min, max, mean, std, NaN counts, Inf counts for each feature."""
    stats = {}
    print("Feature statistics:")
    for i, name in enumerate(feature_names_list):
        feature_col = all_features_tensor[:, i]
        # Filter out NaNs and Infs for meaningful statistics
        valid_mask = ~torch.isnan(feature_col) & ~torch.isinf(feature_col)
        valid_feature_col = feature_col[valid_mask]
        
        current_stats = {
            'nan_count': torch.isnan(feature_col).sum().item(),
            'inf_count': torch.isinf(feature_col).sum().item()
        }

        if valid_feature_col.numel() > 0:
            current_stats['min'] = torch.min(valid_feature_col).item()
            current_stats['max'] = torch.max(valid_feature_col).item()
            current_stats['mean'] = torch.mean(valid_feature_col).item()
            current_stats['std'] = torch.std(valid_feature_col).item()
            
            print("  " + name +
                  ": Min=" + str(current_stats['min']) +
                  ", Max=" + str(current_stats['max']) +
                  ", Mean=" + str(current_stats['mean']) +
                  ", Std=" + str(current_stats['std']) +
                  ", NaNs=" + str(current_stats['nan_count']) +
                  ", Infs=" + str(current_stats['inf_count']))
        else:
            current_stats['min'] = float('nan')
            current_stats['max'] = float('nan')
            current_stats['mean'] = float('nan')
            current_stats['std'] = float('nan')
            print("  " + name +
                  ": All values are NaN/Inf or empty." +
                  " NaNs=" + str(current_stats['nan_count']) +
                  ", Infs=" + str(current_stats['inf_count']))
        stats[name] = current_stats
    return stats

def main():
    r"""Main function to perform data loading, preprocessing, and saving."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created output directory: " + output_dir)

    # 1. Load the dataset
    print("Loading dataset from: " + f_tree)
    try:
        trainset = torch.load(f_tree, map_location=torch.device('cpu'))
    except Exception as e:
        print("Error loading dataset: " + str(e))
        print("This may be due to an issue with the file itself, or if the file contains custom PyTorch Geometric Data objects, ensure the necessary classes are defined/imported.")
        return

    if not isinstance(trainset, list) or not all(hasattr(d, 'x') for d in trainset if d is not None):
        print("Loaded data is not in the expected format (list of Data objects with 'x' attribute).")
        if isinstance(trainset, list) and len(trainset) > 0:
             print("Type of first element: " + str(type(trainset[0])))
        elif isinstance(trainset, list):
             print("Trainset is an empty list.")
        else:
             print("Trainset is not a list. Type: " + str(type(trainset)))
        return
        
    print("Dataset loaded successfully. Number of merger trees: " + str(len(trainset)))
    if len(trainset) > 0 and trainset[0] is not None:
        sample_tree = trainset[0]
        print("Example tree 0: " + str(sample_tree))
        print("  Node features shape (x): " + (str(sample_tree.x.shape) if hasattr(sample_tree, 'x') and sample_tree.x is not None else 'N/A'))
        print("  Target shape (y): " + (str(sample_tree.y.shape) if hasattr(sample_tree, 'y') and sample_tree.y is not None else 'N/A'))
        if hasattr(sample_tree, 'y') and sample_tree.y is not None:
            print("  Target value (y) for tree 0: " + str(sample_tree.y))


    # --- Data Integrity Verification and Preprocessing ---
    all_initial_features_list = []
    valid_trees_indices = [] # Keep track of trees that are valid for processing
    for i, data in enumerate(trainset):
        if data is None:
            print("Warning: Tree " + str(i) + " is None. Skipping.")
            continue
        if not hasattr(data, 'x') or data.x is None:
            print("Warning: Tree " + str(i) + " has no node features 'x'. Skipping.")
            continue
        if data.x.ndim != 2 or data.x.shape[1] != len(feature_names):
            print("Warning: Tree " + str(i) + " has unexpected feature shape " + str(data.x.shape) +
                  ". Expected (*, " + str(len(feature_names)) + "). Skipping.")
            continue
        all_initial_features_list.append(data.x.clone().detach().to(torch.float32))
        valid_trees_indices.append(i)

    if not all_initial_features_list:
        print("No valid node features found in the dataset. Exiting.")
        return
        
    all_initial_features = torch.cat(all_initial_features_list, dim=0)
    total_nodes = all_initial_features.shape[0]
    print("Total number of nodes across " + str(len(all_initial_features_list)) + " valid trees: " + str(total_nodes))

    print("\n--- Initial Feature Statistics (before any processing) ---")
    get_feature_stats(all_initial_features, feature_names)
    
    processed_features_log_list = [] 

    print("\n--- Applying Logarithmic Transformation ---")
    for tree_idx in valid_trees_indices:
        data = trainset[tree_idx]
        current_features = data.x.clone().detach().to(torch.float32) # Already ensured x exists and is float32

        for feature_idx in log_transform_indices:
            feature_name = feature_names[feature_idx]
            column_data = current_features[:, feature_idx]
            
            non_positive_mask = column_data <= 0
            num_non_positive = torch.sum(non_positive_mask).item()
            if num_non_positive > 0:
                print("  Tree " + str(tree_idx) + ": Feature '" + feature_name + "' has " + str(num_non_positive) +
                      " non-positive values (out of " + str(column_data.shape[0]) + "). Clamping to " +
                      str(log_epsilon) + " before log10.")
                column_data = torch.clamp(column_data, min=log_epsilon)
            
            current_features[:, feature_idx] = torch.log10(column_data)

        data.x = current_features 
        processed_features_log_list.append(data.x)

    if not processed_features_log_list:
        print("No features were processed with log transform (e.g., all valid trees skipped or no log_transform_indices). Exiting.")
        return

    all_logged_features = torch.cat(processed_features_log_list, dim=0)
    print("\n--- Feature Statistics (after log transform, before standardization) ---")
    get_feature_stats(all_logged_features, feature_names)

    print("\n--- Standardizing Features ---")
    means = torch.zeros(all_logged_features.shape[1], dtype=torch.float32)
    stds = torch.zeros(all_logged_features.shape[1], dtype=torch.float32)

    for feature_idx in range(all_logged_features.shape[1]):
        col = all_logged_features[:, feature_idx]
        valid_col = col[~torch.isnan(col) & ~torch.isinf(col)]
        if valid_col.numel() > 0:
            means[feature_idx] = torch.mean(valid_col)
            stds[feature_idx] = torch.std(valid_col)
        else: 
            means[feature_idx] = 0.0 
            stds[feature_idx] = 1.0  
            print("Warning: Feature '" + feature_names[feature_idx] + 
                  "' contains only NaNs/Infs or is empty after log transform. Mean set to 0, Std to 1.")

    print("Calculated global means for standardization: " + str(means.tolist()))
    print("Calculated global stds for standardization: " + str(stds.tolist()))

    stds_corrected = stds.clone()
    for idx in range(len(stds_corrected)):
        if stds_corrected[idx] < std_epsilon:
            print("  Note: Std for feature '" + feature_names[idx] + "' was < " + str(std_epsilon) +
                  ". Corrected to 1.0 for standardization. Original std: " + str(stds[idx].item()))
            stds_corrected[idx] = 1.0
    
    for tree_idx in valid_trees_indices:
        data = trainset[tree_idx]
        data.x = (data.x - means) / stds_corrected

    all_standardized_features_list = [trainset[tree_idx].x for tree_idx in valid_trees_indices]
            
    all_standardized_features = torch.cat(all_standardized_features_list, dim=0)
    print("\n--- Feature Statistics (after standardization) ---")
    get_feature_stats(all_standardized_features, feature_names)

    print("\n--- Saving Data ---")
    try:
        torch.save(trainset, preprocessed_data_path)
        print("Preprocessed dataset saved to: " + preprocessed_data_path)
    except Exception as e:
        print("Error saving preprocessed dataset: " + str(e))

    scaling_params = {
        'means': means, 
        'stds': stds_corrected,
        'log_transform_indices': log_transform_indices,
        'feature_names': feature_names,
        'log_epsilon': log_epsilon,
        'std_epsilon': std_epsilon
    }
    try:
        torch.save(scaling_params, scaling_params_path)
        print("Feature scaling parameters saved to: " + scaling_params_path)
    except Exception as e:
        print("Error saving scaling parameters: " + str(e))

    print("\n--- Summary of Preprocessing ---")
    print("1. Loaded " + str(len(trainset)) + " merger tree entries.")
    print("   Processed " + str(len(valid_trees_indices)) + " valid trees.")
    print("2. Verified initial data integrity.")
    log_feature_names_str = ", ".join([feature_names[i] for i in log_transform_indices])
    log_indices_str = ", ".join([str(i) for i in log_transform_indices])
    print("3. Applied log10 transformation to features: " + log_feature_names_str + " (indices " + log_indices_str + ").")
    print("   - Non-positive values were clamped to " + str(log_epsilon) + " before log10.")
    print("4. Standardized all features to zero mean and unit variance using global statistics.")
    print("   - Standard deviations < " + str(std_epsilon) + " were treated as 1.0 during division.")
    print("5. Saved preprocessed data and scaling parameters.")
    
    print("\nFinal check for NaNs/Infs in preprocessed data (data.x for valid trees):")
    total_nans = 0
    total_infs = 0
    for tree_idx in valid_trees_indices:
        data = trainset[tree_idx]
        nans = torch.isnan(data.x).sum().item()
        infs = torch.isinf(data.x).sum().item()
        if nans > 0 or infs > 0:
            print("  Tree " + str(tree_idx) + ": " + str(nans) + " NaNs, " + str(infs) + " Infs")
        total_nans += nans
        total_infs += infs
    print("Total NaNs in final preprocessed features (valid trees): " + str(total_nans))
    print("Total Infs in final preprocessed features (valid trees): " + str(total_infs))
    if total_nans > 0 or total_infs > 0:
        print("Warning: NaNs or Infs are present in the final preprocessed features. This might be due to:")
        print("  - Original data containing NaNs/Infs that were not removed by clamping (e.g. if clamping was insufficient).")
        print("  - Issues during standardization if a feature column was entirely NaN/Inf after log transform.")
        print("These may need to be handled (e.g., imputation) in subsequent steps if they cause issues.")

if __name__ == '__main__':
    main()