# filename: codebase/qtt_feature_engineering.py
import os
import torch
import numpy as np
import tensorly as tl
from tensorly.decomposition import tensor_train

# Configuration
PREPROCESSED_DATA_PATH = 'data/preprocessed_merger_trees.pt'
SUBGRAPH_DATA_PATH_PREFIX = 'data/k_hop_subgraphs_k'
AGGREGATED_QTT_OUTPUT_DIR = 'data'
AGGREGATED_QTT_FEATURES_PATH_PREFIX = os.path.join(AGGREGATED_QTT_OUTPUT_DIR, 'aggregated_qtt_features_k')

K_VALUES = [1, 2, 3]  # K values for k-hop subgraphs
QTT_RANKS = [2, 3]    # Example QTT ranks to experiment with

# Ensure TensorLy uses PyTorch backend
tl.set_backend('pytorch')

def get_qtt_tensor_shape_and_ranks(num_nodes, num_features, qtt_rank_scalar):
    r"""Determines the target shape for QTT and the list of ranks for tensor_train.
    Args:
        num_nodes (int): Number of nodes, must be a power of 2.
        num_features (int): Number of features.
        qtt_rank_scalar (int): The desired internal QTT rank.
    Returns:
        tuple: (tensor_shape, ranks_list)
            tensor_shape (list): Shape of the tensor to be decomposed, e.g., [2,2,2,num_features] or [1,num_features].
            ranks_list (list): List of ranks for tensor_train, e.g., [1, r, r, r, 1].
    """
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive for QTT shape calculation.")
    if num_features <= 0:
        raise ValueError("num_features must be positive for QTT shape calculation.")

    if num_nodes == 1:
        tensor_shape = [1, num_features]
    else:
        n_bits = num_nodes.bit_length() - 1  # num_nodes is a power of 2, e.g., 2^n_bits
        tensor_shape = [2] * n_bits + [num_features]

    num_tensor_dims = len(tensor_shape)
    # ranks_list has length num_tensor_dims + 1
    ranks_list = [1] + [qtt_rank_scalar] * (num_tensor_dims - 1) + [1]
    
    return tensor_shape, ranks_list


def calculate_expected_qtt_feature_size(tensor_shape, ranks_list):
    r"""Calculates the total number of parameters in TT cores for given shape and ranks."""
    if not tensor_shape or not ranks_list:
        return 0
    # Basic check for empty dimensions, though get_qtt_tensor_shape_and_ranks should prevent this.
    for dim_size in tensor_shape:
        if dim_size == 0:
            return 0

    expected_size = 0
    current_rank_in = ranks_list[0]  # Should be 1
    for i in range(len(tensor_shape)):
        dim_i = tensor_shape[i]
        rank_out_i = ranks_list[i+1]
        expected_size += current_rank_in * dim_i * rank_out_i
        current_rank_in = rank_out_i
    return expected_size


def main():
    r"""Main function for QTT decomposition, feature engineering, and aggregation."""
    if not os.path.exists(AGGREGATED_QTT_OUTPUT_DIR):
        os.makedirs(AGGREGATED_QTT_OUTPUT_DIR)
        print("Created output directory: " + AGGREGATED_QTT_OUTPUT_DIR)

    print("Loading preprocessed dataset to get tree_ids and target values: " + PREPROCESSED_DATA_PATH)
    try:
        preprocessed_trainset = torch.load(PREPROCESSED_DATA_PATH, map_location=torch.device('cpu'))
    except Exception as e:
        print("Error loading preprocessed dataset: " + str(e))
        return
    
    num_total_trees = len(preprocessed_trainset)
    print("Total number of trees in dataset: " + str(num_total_trees))
    
    tree_metadata = []
    for i, tree_data_item in enumerate(preprocessed_trainset):
        if tree_data_item is None:  # Handle None entries if any
            print("Warning: Tree " + str(i) + " is None in preprocessed_trainset. Skipping for metadata.")
            lh_id = i  # Placeholder
            y_target = torch.empty((1,0))
        else:
            lh_id = tree_data_item.lh_id if hasattr(tree_data_item, 'lh_id') and tree_data_item.lh_id is not None else i
            y_target = tree_data_item.y if hasattr(tree_data_item, 'y') and tree_data_item.y is not None else torch.empty((1,0))
        tree_metadata.append({'tree_idx': i, 'lh_id': lh_id, 'y': y_target})


    for k_val in K_VALUES:
        subgraph_data_path = SUBGRAPH_DATA_PATH_PREFIX + str(k_val) + '.pt'
        print("\n--- Processing for k = " + str(k_val) + " ---")
        print("Loading subgraph data from: " + subgraph_data_path)
        try:
            subgraphs_for_k = torch.load(subgraph_data_path, map_location=torch.device('cpu'))
        except FileNotFoundError:
            print("Subgraph data file not found: " + subgraph_data_path + ". Skipping k=" + str(k_val) + ".")
            continue
        except Exception as e:
            print("Error loading subgraph data for k=" + str(k_val) + ": " + str(e) + ". Skipping k=" + str(k_val) + ".")
            continue

        if not subgraphs_for_k:
            print("No subgraphs found for k=" + str(k_val) + ". Skipping QTT processing for this k.")
            continue
        
        first_subgraph = subgraphs_for_k[0]
        # Ensure padded_feature_matrix exists and has shape attribute
        if not hasattr(first_subgraph.get('padded_feature_matrix'), 'shape'):
             print("Warning: First subgraph for k=" + str(k_val) + " is missing 'padded_feature_matrix' or it's not a tensor. Skipping k.")
             continue

        fixed_num_nodes = first_subgraph['padded_feature_matrix'].shape[0]
        num_features = first_subgraph['padded_feature_matrix'].shape[1]

        if fixed_num_nodes == 0 or num_features == 0:
            print("Warning: Subgraphs for k=" + str(k_val) + " have zero nodes or zero features (fixed_num_nodes=" + str(fixed_num_nodes) + ", num_features=" + str(num_features) + "). Skipping QTT for this k.")
            continue

        print("  Parameters for k=" + str(k_val) + ": fixed_num_nodes_for_qtt=" + str(fixed_num_nodes) + ", num_features=" + str(num_features))

        for qtt_r in QTT_RANKS:
            print("  -- Processing for QTT rank = " + str(qtt_r) + " --")

            try:
                target_tensor_shape, tt_ranks_list = get_qtt_tensor_shape_and_ranks(fixed_num_nodes, num_features, qtt_r)
                expected_feature_size = calculate_expected_qtt_feature_size(target_tensor_shape, tt_ranks_list)
            except ValueError as e:
                print("    Error in QTT parameter calculation: " + str(e) + ". Skipping this rank for k=" + str(k_val))
                continue
            
            print("    Target tensor shape for TT: " + str(target_tensor_shape))
            print("    TT ranks list: " + str(tt_ranks_list))
            print("    Expected QTT feature vector size: " + str(expected_feature_size))

            if expected_feature_size == 0:
                print("    Warning: Expected QTT feature size is 0. Skipping this rank for k=" + str(k_val))
                continue

            tree_idx_to_qtt_vectors = {i: [] for i in range(num_total_trees)}
            all_reconstruction_errors_mse = []
            num_subgraphs_processed = 0

            for subgraph_info in subgraphs_for_k:
                original_matrix = subgraph_info['padded_feature_matrix'].clone().detach()
                tree_id = subgraph_info['tree_id'] 

                if fixed_num_nodes == 1:
                    tensor_for_qtt = original_matrix 
                else:
                    try:
                        tensor_for_qtt = original_matrix.reshape(target_tensor_shape)
                    except Exception as e:
                        print("Error reshaping matrix for tree_id " + str(tree_id) + ", k=" + str(k_val) + ", rank=" + str(qtt_r) + ". Matrix shape: " + str(original_matrix.shape) + ", Target QTT shape: " + str(target_tensor_shape) + ". Error: " + str(e))
                        continue 
                try:
                    cores = tensor_train(tensor_for_qtt, rank=tt_ranks_list)
                except Exception as e:
                    print("Error in tensor_train for tree_id " + str(tree_id) + ", k=" + str(k_val) + ", rank=" + str(qtt_r) + ". Tensor shape: " + str(tensor_for_qtt.shape) + ", Ranks: " + str(tt_ranks_list) + ". Error: " + str(e))
                    continue 
                
                try:
                    qtt_feature_vector_list = []
                    for core_idx, core in enumerate(cores):
                        if not isinstance(core, torch.Tensor):  # Ensure tensor
                            core = torch.tensor(core, device=original_matrix.device, dtype=original_matrix.dtype)
                        qtt_feature_vector_list.append(core.flatten())
                    qtt_feature_vector = torch.cat(qtt_feature_vector_list)
                except Exception as e:
                    print("Error flattening/concatenating cores for tree_id " + str(tree_id) + ", k=" + str(k_val) + ", rank=" + str(qtt_r) + ". Error: " + str(e))
                    continue

                if qtt_feature_vector.numel() != expected_feature_size:
                    # This case indicates that tensor_train might have returned cores with smaller ranks
                    # than specified if the data was very simple (e.g., all zeros).
                    # We must pad/truncate to ensure all feature vectors have the same fixed size.
                    if qtt_feature_vector.numel() < expected_feature_size:
                        padding = torch.zeros(expected_feature_size - qtt_feature_vector.numel(), 
                                              dtype=qtt_feature_vector.dtype, device=qtt_feature_vector.device)
                        qtt_feature_vector = torch.cat([qtt_feature_vector, padding])
                    else:  # qtt_feature_vector.numel() > expected_feature_size (less likely with fixed ranks)
                        qtt_feature_vector = qtt_feature_vector[:expected_feature_size]

                tree_idx_to_qtt_vectors[tree_id].append(qtt_feature_vector)
                num_subgraphs_processed += 1
                
                try:
                    reconstructed_tensor = tl.tt_to_tensor(cores)
                    reconstructed_matrix = reconstructed_tensor.reshape(original_matrix.shape)
                    mse = torch.mean((original_matrix - reconstructed_matrix)**2)
                    all_reconstruction_errors_mse.append(mse.item())
                except Exception as e:
                    print("Error during reconstruction/MSE calculation for tree_id " + str(tree_id) + ", k=" + str(k_val) + ", rank=" + str(qtt_r) + ". Error: " + str(e))

            if num_subgraphs_processed == 0:
                print("    No subgraphs were successfully processed with QTT for k=" + str(k_val) + ", rank=" + str(qtt_r) + ".")
            else:
                avg_mse = np.mean(all_reconstruction_errors_mse) if all_reconstruction_errors_mse else float('nan')
                print("    Number of subgraphs processed with QTT: " + str(num_subgraphs_processed))
                print("    Average reconstruction MSE: " + str(avg_mse))

            final_aggregated_data = []
            num_trees_with_features = 0
            default_dtype = torch.float32  # Ensure consistent dtype for zero vectors and aggregated vectors

            for i in range(num_total_trees):
                # Check if metadata exists for this index, relevant if preprocessed_trainset had Nones
                if i >= len(tree_metadata): continue 
                meta = tree_metadata[i]
                tree_qtt_list = tree_idx_to_qtt_vectors.get(meta['tree_idx'], [])  # Use .get for safety
                
                aggregated_vector = torch.zeros(expected_feature_size, dtype=default_dtype)
                if tree_qtt_list:
                    try:
                        processed_list = [vec.to(default_dtype) for vec in tree_qtt_list]
                        stacked_vectors = torch.stack(processed_list)
                        aggregated_vector = torch.mean(stacked_vectors, dim=0)
                        num_trees_with_features += 1
                    except RuntimeError as e: 
                        print("    Error stacking QTT vectors for tree_idx " + str(meta['tree_idx']) + ": " + str(e) + ". Using zero vector.")
                    except Exception as e:
                        print("    Unexpected error during aggregation for tree_idx " + str(meta['tree_idx']) + ": " + str(e) + ". Using zero vector.")

                final_aggregated_data.append({
                    'tree_idx': meta['tree_idx'],
                    'lh_id': meta['lh_id'],
                    'y': meta['y'], 
                    'qtt_feature_vector': aggregated_vector,
                    'k_value': k_val,
                    'qtt_rank': qtt_r
                })
            
            print("    Number of trees with actual (non-zero) aggregated QTT features: " + str(num_trees_with_features) + " out of " + str(num_total_trees))

            output_filename = AGGREGATED_QTT_FEATURES_PATH_PREFIX + str(k_val) + '_rank' + str(qtt_r) + '.pt'
            try:
                torch.save(final_aggregated_data, output_filename)
                print("    Aggregated QTT features saved to: " + output_filename)
            except Exception as e:
                print("    Error saving aggregated QTT features: " + str(e))
                
    print("\n--- QTT feature engineering and aggregation complete. ---")


if __name__ == '__main__':
    main()
