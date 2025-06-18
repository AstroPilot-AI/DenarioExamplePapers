# Methodology

## Data Acquisition and Preprocessing

The analysis utilizes a dataset of 500 cosmological merger trees, each representing the hierarchical assembly history of a dark matter halo, stored in PyTorch Geometric format. Each tree contains node features—mass, concentration, vmax, and scale factor—along with edge information and main branch masks. The data is loaded using PyTorch and all subsequent processing is performed on CPUs due to computational constraints.

Prior to feature engineering, node features are preprocessed to ensure numerical stability and comparability:
- **Logarithmic Transformation:** Mass and vmax are log-transformed to reduce dynamic range and approximate normality.
- **Normalization:** All features are standardized to zero mean and unit variance, using statistics computed from the training set. This step is essential for effective tensor decomposition and to prevent any single feature from dominating the representation.

## Subgraph Extraction

For each merger tree, the main branch is identified using the provided mask. To capture the local environment and assembly history, k-hop subgraphs are extracted around each node along the main branch. The k-hop neighborhood includes all nodes reachable within k edges from the central node, as well as the connecting edges. The value of k is selected empirically (e.g., k = 1, 2, 3) to balance local detail with computational tractability. For each tree, this process yields a collection of overlapping subgraphs, each represented by its node feature matrix.

## QTT-Based Feature Engineering

Each subgraph’s node feature matrix (shape: [number of nodes, 4]) is prepared for Quantum Tensor Train (QTT) decomposition. If necessary, matrices are padded or reshaped to fit the requirements of the QTT algorithm (e.g., to a power-of-two size). QTT decomposition is then applied to each matrix, producing a sequence of low-rank tensor cores. These cores are flattened and concatenated to form a fixed-length QTT-based feature vector for each subgraph. The QTT rank is chosen based on computational feasibility and the desired level of compression, with lower ranks yielding more aggressive compression.

## Feature Aggregation

Since each merger tree yields multiple QTT feature vectors (one per main branch node), these are aggregated to produce a single feature vector per tree. Aggregation strategies include:
- **Mean pooling:** Averaging QTT vectors across all subgraphs.
- **Max pooling:** Taking the element-wise maximum.
- **Concatenation:** Concatenating QTT vectors from a fixed number of main branch nodes (e.g., the last n nodes).

The aggregation method is selected based on empirical performance and interpretability.

## Regression Modeling

The aggregated QTT-based feature vectors serve as input to a regression model tasked with predicting the final halo mass at z=0 (the mass of the main progenitor at the final snapshot). Given the moderate dataset size and CPU-only environment, ensemble tree-based models such as Random Forests or Gradient Boosting Machines are employed for their robustness and interpretability. Model hyperparameters are optimized via cross-validation. Baseline models using traditional features (e.g., mean, max, or variance of node features along the main branch) are also trained for comparison.

Model performance is evaluated using:
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **Coefficient of Determination (R²)**

## Visualization and Interpretation

A comprehensive suite of visualizations is produced to assess the effectiveness and interpretability of QTT-based features:
- **Feature Distributions:** Histograms and density plots of QTT features, compared to traditional features.
- **Regression Performance:** Scatter plots of predicted vs. true halo mass, residual plots, and bar charts of performance metrics for QTT-based and baseline models.
- **Feature Importance:** Rankings of QTT feature importances from tree-based models, and correlation heatmaps between QTT features and the target variable.
- **Physical Interpretation:** Visualization of selected subgraphs with QTT feature overlays, and dimensionality reduction (e.g., PCA or t-SNE) of QTT feature space colored by halo mass.

## Addressing Implementation Challenges

Several challenges were encountered and addressed during implementation:
- **Computational Constraints:** QTT was applied only to small, localized subgraphs to ensure tractability on CPUs. Efficient batching and memory management were employed.
- **Feature Matrix Size Variability:** Subgraphs of varying sizes were padded or truncated to a fixed size compatible with QTT decomposition.
- **Hyperparameter Selection:** The choice of k (subgraph size), QTT rank, and aggregation method was guided by cross-validation and empirical performance.
- **Interpretability:** Visualization and feature importance analysis were used to interpret the physical meaning of QTT-compressed features.

This methodology enables efficient, interpretable, and physically motivated feature engineering for merger tree data, leveraging QTT to extract salient information for predictive modeling of halo properties.