# filename: codebase/subgraph_extraction.py
import torch
import os
import numpy as np
from torch_geometric.utils import k_hop_subgraph, to_undirected

# Configuration
PREPROCESSED_DATA_PATH = 'data/preprocessed_merger_trees.pt'
OUTPUT_DIR = 'data'
# Using .pt extension for PyTorch serialized objects; using string concatenation instead of format
SUBGRAPH_DATA_PATH_PREFIX = os.path.join(OUTPUT_DIR, 'k_hop_subgraphs_k')
K_VALUES = [1, 2, 3]
PADDING_VALUE = 0.0  # Standardized data, mean is 0


def next_power_of_2(n):
    r"""Computes the next power of 2 greater than or equal to n."""
    if n <= 0:  # Handles n=0 and negative n, though n should be non-negative count
        return 1  # Smallest practical dimension for a tensor array
    # Efficient way to find next power of 2
    return 1 << (n - 1).bit_length()


def pad_or_truncate_features(features_tensor, target_num_nodes, num_node_features_fallback, padding_val=0.0):
    r"""
    Pads or truncates a node feature tensor to a target number of nodes.
    Args:
        features_tensor (torch.Tensor): The input feature tensor of shape [current_num_nodes, num_features].
        target_num_nodes (int): The desired number of nodes.
        num_node_features_fallback (int): Fallback for number of features if features_tensor is completely empty.
        padding_val (float): Value to use for padding.
    Returns:
        torch.Tensor: The processed feature tensor of shape [target_num_nodes, num_features].
    """
    current_num_nodes = features_tensor.shape[0]
    
    # Determine number of features, handling empty tensors
    if features_tensor.nelement() == 0 and current_num_nodes == 0:
        num_features = num_node_features_fallback
    elif features_tensor.ndim < 2:  # e.g. shape (0,) when expecting (0, N_feat)
        num_features = num_node_features_fallback
        # If features_tensor was (0,), reshape to (0, N_feat) to make padding logic simpler
        features_tensor = features_tensor.reshape(0, num_features)
    else:
        num_features = features_tensor.shape[1]

    if current_num_nodes == target_num_nodes:
        return features_tensor
    elif current_num_nodes < target_num_nodes:
        padding_shape = (target_num_nodes - current_num_nodes, num_features)
        padding = torch.full(padding_shape, padding_val, dtype=features_tensor.dtype, device=features_tensor.device)
        if current_num_nodes == 0:
             return padding
        return torch.cat([features_tensor, padding], dim=0)
    else:  # current_num_nodes > target_num_nodes
        return features_tensor[:target_num_nodes, :]


def extract_subgraphs_and_prepare_for_qtt():
    r"""
    Extracts k-hop subgraphs for main branch nodes using an undirected interpretation of edges,
    documents their statistics, pads/truncates feature matrices to a fixed size (power of 2),
    and saves them.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print("Created output directory: " + OUTPUT_DIR)

    print("Loading preprocessed dataset from: " + PREPROCESSED_DATA_PATH)
    try:
        preprocessed_trainset = torch.load(PREPROCESSED_DATA_PATH, map_location=torch.device('cpu'))
    except Exception as e:
        print("Error loading preprocessed dataset: " + str(e))
        return

    if not isinstance(preprocessed_trainset, list) or not preprocessed_trainset:
        print("Loaded data is not a non-empty list. Exiting.")
        return
    
    num_node_features = -1
    first_valid_tree_for_features = next((tree for tree in preprocessed_trainset if 
                                          tree is not None and 
                                          hasattr(tree, 'x') and 
                                          tree.x is not None and 
                                          tree.x.ndim == 2 and 
                                          tree.x.shape[0] > 0), None)
    if first_valid_tree_for_features is not None:
        num_node_features = first_valid_tree_for_features.x.shape[1]
    else:
        first_tree_with_x = next((tree for tree in preprocessed_trainset if
                                  tree is not None and 
                                  hasattr(tree, 'x') and 
                                  tree.x is not None and 
                                  tree.x.ndim == 2), None)
        if first_tree_with_x is not None:
             num_node_features = first_tree_with_x.x.shape[1]
        else:
            print("No valid trees with 2D feature matrix 'x' found. Exiting.")
            return
    
    print("Number of node features determined from data: " + str(num_node_features))

    for k_val in K_VALUES:
        print("\n--- Processing for k = " + str(k_val) + " ---")
        
        all_subgraphs_raw_info = []
        subgraph_node_counts = []
        subgraph_edge_counts = []
        max_nodes_in_any_subgraph_for_k = 0
        processed_tree_count = 0
        skipped_empty_mask_main_count = 0
        skipped_invalid_node_idx_count = 0

        print("Pass 1: Extracting subgraphs and collecting size statistics for k=" + str(k_val) + "...")
        for tree_idx, tree_data in enumerate(preprocessed_trainset):
            if not (tree_data and hasattr(tree_data, 'x') and tree_data.x is not None and \
                    hasattr(tree_data, 'edge_index') and tree_data.edge_index is not None and \
                    hasattr(tree_data, 'mask_main') and tree_data.mask_main is not None and \
                    hasattr(tree_data, 'num_nodes')):
                continue

            if isinstance(tree_data.mask_main, np.ndarray):
                tree_data.mask_main = torch.tensor(tree_data.mask_main, dtype=torch.long)
            
            if tree_data.mask_main.numel() == 0:
                skipped_empty_mask_main_count += 1
                continue
            
            processed_tree_count += 1
            
            if tree_data.edge_index.numel() > 0:
                effective_edge_index = to_undirected(tree_data.edge_index, num_nodes=tree_data.num_nodes)
            else:
                effective_edge_index = tree_data.edge_index

            main_branch_node_indices = tree_data.mask_main
            
            for mb_node_original_idx_tensor in main_branch_node_indices:
                mb_node_original_idx = mb_node_original_idx_tensor.item()

                if not (0 <= mb_node_original_idx < tree_data.num_nodes):
                    skipped_invalid_node_idx_count += 1
                    continue 

                subset, sub_edge_index, central_node_relabelled_idx_tensor, edge_mask = k_hop_subgraph(
                    node_idx=mb_node_original_idx,
                    num_hops=k_val,
                    edge_index=effective_edge_index, 
                    relabel_nodes=True,
                    num_nodes=tree_data.num_nodes 
                )
                
                current_subgraph_num_nodes = subset.shape[0]
                current_subgraph_num_edges = sub_edge_index.shape[1]

                subgraph_node_features = torch.empty((0, num_node_features), dtype=tree_data.x.dtype, device=tree_data.x.device)
                central_node_relabelled_idx = -1 

                if current_subgraph_num_nodes > 0:
                    subgraph_node_features = tree_data.x[subset]
                    if central_node_relabelled_idx_tensor.numel() > 0:
                         central_node_relabelled_idx = central_node_relabelled_idx_tensor.item()
                
                all_subgraphs_raw_info.append({
                    'tree_id': tree_idx,
                    'main_branch_node_original_idx': mb_node_original_idx,
                    'subgraph_node_features_original': subgraph_node_features,
                    'subgraph_edge_index_undirected_relabelled': sub_edge_index, 
                    'subgraph_original_node_indices': subset,
                    'central_node_relabelled_idx': central_node_relabelled_idx,
                    'num_nodes_in_original_subgraph': current_subgraph_num_nodes,
                    'num_edges_in_original_subgraph': current_subgraph_num_edges
                })
                subgraph_node_counts.append(current_subgraph_num_nodes)
                subgraph_edge_counts.append(current_subgraph_num_edges)
                if current_subgraph_num_nodes > max_nodes_in_any_subgraph_for_k:
                    max_nodes_in_any_subgraph_for_k = current_subgraph_num_nodes
        
        print("Pass 1 for k=" + str(k_val) + " completed.")
        print("  Processed " + str(processed_tree_count) + " trees with valid structure and non-empty mask_main.")
        if skipped_empty_mask_main_count > 0:
            print("  Skipped " + str(skipped_empty_mask_main_count) + " trees due to empty mask_main.")
        if skipped_invalid_node_idx_count > 0:
            print("  Skipped " + str(skipped_invalid_node_idx_count) + " main branch nodes due to invalid index.")

        if not all_subgraphs_raw_info:
            print("No subgraphs could be extracted for k=" + str(k_val) + ". Skipping further processing for this k.")
            continue

        fixed_num_nodes_for_qtt = next_power_of_2(max_nodes_in_any_subgraph_for_k)
        print("Max nodes in any original subgraph for k=" + str(k_val) + ": " + str(max_nodes_in_any_subgraph_for_k))
        print("Target fixed_num_nodes for QTT (power of 2) for k=" + str(k_val) + ": " + str(fixed_num_nodes_for_qtt))

        print("\nSubgraph Statistics for k = " + str(k_val) + ":")
        print("  Total subgraphs extracted: " + str(len(subgraph_node_counts)))
        if subgraph_node_counts:
            counts_arr = np.array(subgraph_node_counts)
            print("  Nodes per subgraph (original):")
            print("    Avg: " + str(np.mean(counts_arr)) + ", Std: " + str(np.std(counts_arr)))
            print("    Min: " + str(np.min(counts_arr)) + ", Max: " + str(np.max(counts_arr)))
        if subgraph_edge_counts:
            counts_arr = np.array(subgraph_edge_counts)
            print("  Edges per subgraph (original, from undirected):")
            print("    Avg: " + str(np.mean(counts_arr)) + ", Std: " + str(np.std(counts_arr)))
            print("    Min: " + str(np.min(counts_arr)) + ", Max: " + str(np.max(counts_arr)))
        
        print("Pass 2: Padding/truncating feature matrices for k=" + str(k_val) + "...")
        final_subgraph_data_for_k = []
        for raw_info in all_subgraphs_raw_info:
            padded_feature_matrix = pad_or_truncate_features(
                raw_info['subgraph_node_features_original'],
                fixed_num_nodes_for_qtt,
                num_node_features, 
                PADDING_VALUE
            )
            
            final_subgraph_data_for_k.append({
                'tree_id': raw_info['tree_id'],
                'main_branch_node_original_idx': raw_info['main_branch_node_original_idx'],
                'padded_feature_matrix': padded_feature_matrix,
                'subgraph_edge_index': raw_info['subgraph_edge_index_undirected_relabelled'], 
                'subgraph_original_node_indices': raw_info['subgraph_original_node_indices'],
                'num_nodes_in_original_subgraph': raw_info['num_nodes_in_original_subgraph'],
                'num_edges_in_original_subgraph': raw_info['num_edges_in_original_subgraph'],
                'central_node_relabelled_idx': raw_info['central_node_relabelled_idx'],
                'k_value': k_val,
                'fixed_num_nodes_for_qtt': fixed_num_nodes_for_qtt
            })

        current_k_output_path = SUBGRAPH_DATA_PATH_PREFIX + str(k_val) + '.pt'
        try:
            torch.save(final_subgraph_data_for_k, current_k_output_path)
            print("Saved " + str(len(final_subgraph_data_for_k)) + " processed subgraphs for k=" + str(k_val) + " to: " + current_k_output_path)
        except Exception as e:
            print("Error saving subgraph data for k=" + str(k_val) + ": " + str(e))

    print("\n--- Subgraph extraction and preparation complete. ---")


if __name__ == '__main__':
    extract_subgraphs_and_prepare_for_qtt()