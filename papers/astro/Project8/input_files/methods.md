**Data Preprocessing:**

1.  **Data Loading:** Load the merger tree dataset using `torch.load(f_tree, weights_only=False)`.

2.  **Feature Extraction:** For each merger tree in the dataset, extract the node features (`x`), edge indices (`edge_index`), edge attributes (`edge_attr`), and target variable (`y`). Also extract `node_halo_id` to compute the assembly bias proxy.

3.  **Log Transformation:** Apply a logarithmic transformation to the halo mass and Vmax node features to reduce skewness and improve the distribution of the data. Specifically, replace `x[:, 0]` (mass) and `x[:, 2]` (Vmax) with `torch.log10(x[:, 0])` and `torch.log10(x[:, 2])`, respectively.

4.  **Normalization:** Normalize the log-transformed mass, log-transformed Vmax, and scale factor node features to a range between 0 and 1. This involves computing the minimum and maximum values for each feature across the entire dataset and then scaling each value using the formula: `(value - min) / (max - min)`. Store the min/max values for each feature.

5.  **Edge Feature Engineering:** The edge features are: mass ratio of merging halos, normalized relative velocity difference, and time difference between mergers.
    *   Mass ratio: For each edge, calculate the mass ratio of the child halo to the parent halo, where the parent halo is defined as the halo at the source node of the edge and the child halo is defined as the halo at the target node of the edge. Since we do not have the parent/child information, we will consider the halo with the largest mass as the parent.
    *   Relative velocity difference: We do not have the velocity of the halos, so we will skip this feature.
    *   Time difference: Calculate the time difference between the scale factors of the merging halos.
    Store these values in the `edge_attr` tensor.
    Normalize the edge features to a range between 0 and 1 using the same procedure as for the node features. Store the min/max values for each feature.

6.  **Assembly Bias Proxy Calculation:** Compute an assembly bias proxy for each merger tree. This requires using the `node_halo_id` to identify halos that exist at z=0 (scale factor = 1). For each tree, determine the index of the nodes with a scale factor closest to 1. Note that there can be multiple nodes with a scale factor close to 1. The assembly bias proxy is the mean halo mass of the main halos at z=0.

7.  **Data Splitting:** Divide the dataset into training, validation, and test sets with a ratio of 70:15:15. Ensure that the split is done randomly.

**Topological Data Analysis (TDA):**

1.  **Filtration:** Convert each merger tree into a simplicial complex. Use the scale factor as the filtration parameter. This means that nodes are added to the complex in order of increasing scale factor. Edges are added when both nodes connected by the edge are present in the complex.

2.  **Persistent Homology:** Compute the persistent homology of the simplicial complex using a library like `GUDHI` or `ripser.py`. Calculate the persistence diagrams for H0 (connected components) and H1 (loops).

3.  **Feature Extraction from Persistence Diagrams:** Extract the following topological features from the persistence diagrams:
    *   Number of connected components at the beginning of the filtration.
    *   Number of loops at different scale factor thresholds (e.g., 0.25, 0.5, 0.75, 1.0).
    *   Average persistence of H0 features (death - birth).
    *   Maximum persistence of H0 features.
    *   Average persistence of H1 features (death - birth).
    *   Maximum persistence of H1 features.
    *   Betti numbers: β0 (number of connected components) and β1 (number of loops) at different scale factor thresholds (e.g., 0.25, 0.5, 0.75, 1.0).

4.  **Correlation Analysis:** Calculate the Pearson correlation coefficient between the topological features and the assembly bias proxy on the training set. This will help identify the topological features that are most strongly correlated with assembly bias.

**Graph Neural Network (GNN) Model:**

1.  **GNN Architecture:** Use a Graph Convolutional Network (GCN) architecture. The GNN will consist of multiple GCN layers, followed by a readout layer and a linear regression layer.

    *   **GCN Layers:** Each GCN layer will perform message passing and node feature aggregation. The number of GCN layers will be a hyperparameter to be tuned (e.g., 2, 4, or 6 layers). The hidden dimension of each GCN layer will also be a hyperparameter (e.g., 32, 64, or 128).
    *   **Readout Layer:** After the GCN layers, a readout layer will aggregate the node embeddings into a single graph-level embedding. Use a global mean pooling readout layer.
    *   **Linear Regression Layer:** The graph-level embedding will be passed through a linear regression layer to predict the assembly bias proxy. The output will be a single scalar value.

2.  **GNN Implementation:** Implement the GNN model using PyTorch Geometric. Use the `torch_geometric.nn.GCNConv` module for the GCN layers.

3.  **Loss Function:** Use the Mean Squared Error (MSE) loss function to train the GNN.

4.  **Optimizer:** Use the Adam optimizer to update the GNN weights. The learning rate and weight decay will be hyperparameters to be tuned.

**Training and Evaluation:**

1.  **Hyperparameter Tuning:** Perform hyperparameter tuning using the validation set. The hyperparameters to be tuned include:
    *   Number of GCN layers
    *   Hidden dimension of GCN layers
    *   Learning rate
    *   Weight decay
    *   Batch size
    *   Number of epochs
    Use a grid search or random search to explore the hyperparameter space.

2.  **Training Loop:** Train the GNN model on the training set for a fixed number of epochs. In each epoch, iterate over the training data in batches. For each batch, compute the loss, calculate the gradients, and update the GNN weights using the Adam optimizer.

3.  **Validation:** After each epoch, evaluate the GNN model on the validation set. Compute the MSE loss and the R-squared score. Use the validation set to select the best GNN model based on the validation loss.

4.  **Testing:** After training, evaluate the best GNN model on the test set. Compute the MSE loss and the R-squared score. These metrics will provide an estimate of the GNN model's generalization performance.

**Workflow Summary:**

1.  Load and preprocess the merger tree data, including log transformation, normalization, edge feature engineering, and assembly bias proxy calculation.
2.  Split the data into training, validation, and test sets.
3.  Perform TDA on the training set to extract topological features from the merger trees.
4.  Calculate the correlation between the topological features and the assembly bias proxy.
5.  Design and implement a GNN model using PyTorch Geometric.
6.  Train the GNN model on the training set, using the validation set for hyperparameter tuning and model selection.
7.  Evaluate the best GNN model on the test set.