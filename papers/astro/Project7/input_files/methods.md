This document outlines the methodology for investigating the relationship between dark matter halo merger tree morphology and cosmological parameters (Omega\_m, sigma\_8) using graph spectral analysis and diffusion geometry.

**I. Data Loading and Preprocessing**

A. **Data Loading**
The dataset consists of 1000 merger trees from N-body simulations, stored in a PyTorch Geometric format.
python
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix, get_laplacian, to_networkx
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
# For GCN baseline
# from torch_geometric.nn import GCNConv, global_mean_pool
# import torch.nn.functional as F

# Load the dataset
f_tree = '/Users/fvillaescusa/Documents/Software/AstroPilot/data/Pablo_merger_trees2.pt'
trainset = torch.load(f_tree, weights_only=False) # List of 1000 PyG Data objects

Each `Data` object contains:
- `x`: Node features `[num_nodes, 4]` (log10(mass), log10(concentration), log10(Vmax), scale factor).
- `edge_index`: Graph connectivity `[2, num_edges]`.
- `edge_attr`: Edge features `[num_edges, 1]`. We will investigate the nature of this feature. If it's not directly usable for our astrophysically motivated features, we will compute them from node features.
- `y`: Graph-level labels `[1, 2]` (Omega\_m, sigma\_8).
- `num_nodes`: Number of nodes in the graph.
- `lh_id`: Latin Hypercube ID, identifying the simulation (25 trees per simulation).
- `node_halo_id`: Halo identifiers for nodes.

B. **Exploratory Data Analysis (EDA)**
Before feature engineering, conduct a thorough EDA to understand data characteristics.

1.  **Node Feature Distributions:**
    Analyze the distributions of the four node features across all nodes in all 1000 trees.
    *Hypothetical EDA Key Statistics (to be verified with actual data):*
    | Feature             | Mean  | Std   | Min   | Max   | Notes                               |
    |---------------------|-------|-------|-------|-------|-------------------------------------|
    | log10(mass)         | 12.2  | 1.1   | 10.0  | 15.0  | Halo mass in M_sun/h                |
    | log10(concentration)| 0.75  | 0.25  | 0.2   | 1.4   | Halo concentration                  |
    | log10(Vmax)         | 2.4   | 0.4   | 1.8   | 3.2   | Max circular velocity in km/s       |
    | scale factor        | 0.65  | 0.22  | 0.08  | 1.0   | Cosmological scale factor (a)       |
    *Rationale:* Understanding these distributions and ranges is crucial for normalization and identifying potential outliers or issues.

2.  **Graph Target Variable Distributions:**
    Analyze the distributions of Omega\_m and sigma\_8. There are 1000 trees from 40 unique simulations (1000 trees / 25 trees/simulation = 40 simulations). The `y` values are constant for all trees from the same `lh_id`.
    *Hypothetical EDA Key Statistics (to be verified with actual data for the 40 unique (Omega_m, sigma_8) pairs):*
    | Parameter | Mean  | Std   | Min   | Max   |
    |-----------|-------|-------|-------|-------|
    | Omega\_m  | 0.30  | 0.12  | 0.1   | 0.5   |
    | sigma\_8  | 0.80  | 0.09  | 0.6   | 1.0   |
    *Rationale:* This confirms the target ranges and variability we aim to predict.

3.  **Graph Structural Properties:**
    Analyze the distribution of graph sizes.
    *Hypothetical EDA Key Statistics (to be verified with actual data for 1000 graphs):*
    | Property        | Mean | Std | Min | Max  | Notes                                           |
    |-----------------|------|-----|-----|------|-------------------------------------------------|
    | Number of Nodes | 300  | 150 | 30  | 800  | Significant variability in tree size.           |
    | Number of Edges | 299  | 150 | 29  | 799  | For trees, num_edges = num_nodes - 1.           |
    *Rationale:* The variability in graph size necessitates graph embedding methods that produce fixed-size representations (e.g., spectral moments, aggregated diffusion embeddings).

4.  **Edge Attribute (`edge_attr`) Investigation:**
    Examine the nature and distribution of the provided `edge_attr` (shape `[num_edges, 1]`). If this attribute is not directly interpretable as a physical quantity relevant to merger events (like mass ratio or scale factor difference), we will proceed to compute these features manually as described in section II.C.

C. **Data Splitting (Simulation-aware)**
The dataset will be split into training (e.g., 80%) and testing (e.g., 20%) sets. Crucially, this split must be performed at the simulation level using the `lh_id` to prevent data leakage, as all 25 trees from one simulation share the same `y` values.
python
# Example splitting logic
all_lh_ids = np.array([data.lh_id.item() for data in trainset])
unique_lh_ids = np.unique(all_lh_ids)
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42) # For 40 simulations, this means 32 train, 8 test sims
train_sim_idx, test_sim_idx = next(splitter.split(unique_lh_ids, groups=unique_lh_ids))
train_simulation_ids = unique_lh_ids[train_sim_idx]
test_simulation_ids = unique_lh_ids[test_sim_idx]

train_indices = [i for i, lh_id in enumerate(all_lh_ids) if lh_id in train_simulation_ids]
test_indices = [i for i, lh_id in enumerate(all_lh_ids) if lh_id in test_simulation_ids]

# train_graphs = [trainset[i] for i in train_indices]
# test_graphs = [trainset[i] for i in test_indices]


D. **Feature Normalization (Node features)**
Node features (`x`) will be normalized. Calculate the mean and standard deviation for each of the 4 node features *only from the training set graphs*. Then apply this transformation to both training and testing sets.
python
# Placeholder for normalization logic
# Concatenate all node features from training graphs
# all_train_x = torch.cat([graph.x for graph in train_graphs], dim=0)
# scaler_x = StandardScaler()
# scaler_x.fit(all_train_x.numpy())
# For each graph: graph.x = torch.tensor(scaler_x.transform(graph.x.numpy()), dtype=torch.float)

Target variables `y` (Omega\_m, sigma\_8) are typically not normalized before feeding into tree-based regressors like Random Forest or Gradient Boosting, but we will monitor their distributions.

**II. Feature Engineering from Merger Trees**

For each graph in the dataset, we will extract three sets of features:

A. **Graph Spectral Feature Extraction**
These features capture the global connectivity and structure of the merger trees.

1.  **Normalized Graph Laplacian Computation:**
    For each graph `g = (V, E)`:
    Convert `g.edge_index` to a SciPy sparse adjacency matrix `A`.
    Compute the normalized graph Laplacian: `L_norm = I - D^(-1/2) A D^(-1/2)`, where `D` is the diagonal degree matrix and `I` is the identity matrix.
    *Rationale:* `L_norm` has eigenvalues in `[0, 2]`, providing a stable basis for spectral analysis.

2.  **Eigenvalue Decomposition:**
    Compute the eigenvalues of `L_norm`. Since graph sizes vary, storing all eigenvalues is problematic.
    *Rationale:* Eigenvalues of the Laplacian (the spectrum) are graph invariants and reflect structural properties.

3.  **Spectral Moment Calculation:**
    To obtain a fixed-size feature vector for each graph, compute spectral moments from its eigenvalue distribution (e.g., the first `k=5` moments):
    -   Mean of eigenvalues
    -   Standard deviation of eigenvalues
    -   Skewness of eigenvalues
    -   Kurtosis of eigenvalues
    -   Sum of the smallest `m` non-zero eigenvalues (e.g., `m=10`, if available, padded if fewer).
    Alternatively, consider a fixed number of smallest non-zero eigenvalues if graphs are large enough, or use them to compute summary statistics. We will prioritize spectral moments for fixed-size vectors.
    *Rationale:* Spectral moments summarize the shape of the eigenvalue distribution, providing a compact structural signature.

B. **Diffusion Geometry Embedding**
This approach embeds nodes in a low-dimensional space that captures diffusion dynamics on the graph.

1.  **Diffusion Operator Construction:**
    For each graph, construct a diffusion operator. We will use the random walk transition matrix `P = D^(-1)A`.
    *Rationale:* `P` describes the probability of transitioning between connected nodes in one step of a random walk. Its eigenvectors reveal clusters and geometric properties related to diffusion.

2.  **Eigenvector Computation for Node Embeddings:**
    Compute the eigenvectors of `P`. The eigenvectors corresponding to the largest eigenvalues (closest to 1) will be used.
    Select the top `d` eigenvectors (e.g., `d=3` to `d=5`, excluding the trivial eigenvector for eigenvalue 1 if `P` is irreducible) to form a `d`-dimensional embedding for each node in the graph.
    *Rationale:* These eigenvectors provide a low-dimensional representation of nodes reflecting their position within the graph's diffusion geometry.

3.  **Graph-level Aggregation of Node Embeddings:**
    Aggregate these `d`-dimensional node embeddings into a fixed-size graph-level feature vector using:
    -   Mean pooling (average of all node embeddings)
    -   Max pooling (element-wise maximum)
    -   Min pooling (element-wise minimum)
    Concatenate these aggregated vectors. For `d=3`, this would yield `3 (mean) + 3 (max) + 3 (min) = 9` features.
    *Rationale:* Aggregation summarizes the distribution of node positions in the diffusion embedding space for the entire graph.

C. **Astrophysically-Motivated Edge Feature Aggregation**
These features aim to capture characteristics of merger events.

1.  **Calculation of Edge-level Physical Properties:**
    For each edge `(u, v)` in a merger tree (representing a progenitor `u` merging into/becoming descendant `v`):
    -   **Scale Factor Difference:** `delta_sf = |scale_factor(u) - scale_factor(v)|`. Ensure consistent interpretation of edge direction (progenitor to descendant means `scale_factor(v) > scale_factor(u)`).
    -   **Log Mass Ratio:** `log_mass_ratio = log10(mass(v)) - log10(mass(u))` (descendant mass / progenitor mass). If edges point from progenitor to descendant, this should generally be positive or zero. Alternatively, use `log10(mass_progenitor / mass_descendant)`. We will clarify edge direction from data structure (typically edges point from past to future/progenitor to descendant). For calculation, we'll retrieve node features `x[u, feature_idx]` and `x[v, feature_idx]`.
    *Rationale:* These quantities directly probe the physical nature of halo mergers, such as their timing and mass accretion.

2.  **Aggregation to Graph-level Statistics:**
    For each graph, calculate the following statistics for the distribution of `delta_sf` and `log_mass_ratio` over all its edges:
    -   Mean
    -   Variance
    This will yield 4 features per graph (mean/variance for `delta_sf`, mean/variance for `log_mass_ratio`).
    *Rationale:* These statistics summarize the typical and an-typical merger event characteristics within a tree.

**III. Model Development and Evaluation**

A. **Feature Vector Assembly and Dimensionality Reduction**

1.  **Concatenation of Engineered Features:**
    For each graph, concatenate the features derived from:
    -   Graph spectral moments (Section II.A.3)
    -   Aggregated diffusion map embeddings (Section II.B.3)
    -   Aggregated astrophysically-motivated edge statistics (Section II.C.2)
    This results in a single feature vector per merger tree.

2.  **Principal Component Analysis (PCA):**
    Apply PCA to the combined feature vectors derived from the *training set*. Determine the number of principal components to retain (e.g., explaining 95% or 99% of the variance). Transform both training and testing set feature vectors using the fitted PCA.
    *Rationale:* PCA reduces dimensionality, decorrelates features, and can help improve the performance and stability of downstream regression models.

B. **Regression Models for Cosmological Parameter Prediction**
We will train separate regression models to predict Omega\_m and sigma\_8, or a multi-output regressor if supported well.

1.  **Model Selection:**
    -   Random Forest Regressor
    -   Gradient Boosting Regressor (e.g., XGBoost, LightGBM)
    *Rationale:* These are robust, non-linear models that perform well on tabular data and can capture complex relationships. They are also less computationally expensive than deep neural networks for this scale of data.

2.  **Training and Hyperparameter Optimization:**
    Train the selected models on the PCA-transformed feature vectors from the training set.
    Perform hyperparameter optimization for each model using cross-validation (e.g., `GridSearchCV` or `RandomizedSearchCV`) on the training set. The cross-validation splits should also respect the `lh_id` groupings to prevent leakage.

C. **Evaluation Metrics**
Evaluate the performance of the trained models on the *test set* using:
-   **R-squared (R²)**: Coefficient of determination.
-   **Mean Squared Error (MSE)**.
-   **Uncertainty Calibration Metrics:**
    -   **Expected Calibration Error (ECE):** If models provide uncertainty estimates (e.g., quantile regression for GBTs, or variance from RFs), assess their calibration.
    -   **Reliability Diagrams:** Visually inspect calibration by plotting predicted probabilities/quantiles against observed frequencies.
    *Rationale:* A comprehensive set of metrics is needed to assess predictive accuracy and the reliability of any uncertainty estimates.

D. **Baseline Models for Comparison**
To contextualize the performance of our proposed feature set, we will compare against two baselines:

1.  **Baseline 1: Aggregated Node Features with Classical Regressors**
    -   **Feature Engineering:** For each graph, compute simple aggregated node features: mean, standard deviation, min, and max for each of the 4 raw node features (log10(mass), log10(concentration), log10(Vmax), scale factor). This yields a 4 features * 4 aggregators = 16-dimensional feature vector per graph. Node features should be normalized as per I.D before aggregation.
    -   **Modeling:** Train Random Forest and Gradient Boosting regressors on these aggregated features, using the same training/testing split and hyperparameter optimization strategy.
    -   **Evaluation:** Evaluate using the same metrics (R², MSE).
    *Rationale:* This baseline tests whether sophisticated graph features outperform simple global statistics of node properties.

2.  **Baseline 2: Graph Convolutional Network (GCN)**
    -   **Model Architecture:** Implement a simple GCN for graph-level regression.
        -   Input: Normalized node features `x`.
        -   Layers: e.g., 2-3 `GCNConv` layers with ReLU activations.
        -   Pooling: A global mean pooling layer to obtain a graph-level embedding.
        -   Output: An MLP (e.g., 2 fully connected layers) to regress Omega\_m and sigma\_8.
    -   **Training:** Train the GCN on the training graph data, using an appropriate loss function (e.g., MSE). Training will be performed on CPUs.
    -   **Evaluation:** Evaluate on the test set using R² and MSE.
    *Rationale:* This baseline compares our "classical" graph feature engineering approach against a common geometric deep learning model.

This detailed methodology will guide the systematic extraction of features from merger trees and the subsequent modeling and evaluation process to unveil cosmological signatures.