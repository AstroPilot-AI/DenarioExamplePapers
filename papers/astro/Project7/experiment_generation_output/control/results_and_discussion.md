<!-- filename: results_and_discussion.md -->
## Results and Discussion

This section details the outcomes of applying graph spectral analysis, diffusion geometry, and baseline machine learning techniques to predict cosmological parameters ($\Omega_m$ and $\sigma_8$) from dark matter halo merger trees. We evaluate the efficacy of engineered graph features against simpler node aggregation methods and a Graph Convolutional Network (GCN) baseline.

### Data Preprocessing and Feature Engineering

The dataset comprised 1000 merger trees, originating from 40 unique N-body simulations (25 trees per simulation). Node features (log10(mass), log10(concentration), log10(Vmax), scale factor) were normalized using mean and standard deviation derived from the training set (800 trees from 32 simulations). The normalization parameters were: mean = `[11.138, 0.736, 2.115, 0.370]` and std = `[0.713, 0.364, 0.212, 0.180]` for the four node features, respectively (details in `data/normalization_params.npz`).

A suite of 24 engineered features was constructed for each graph:
1.  **Edge-based features (4):** Mean and variance of the absolute difference in scale factors between connected nodes, and mean and variance of the log-ratio of descendant mass to progenitor mass.
2.  **Laplacian spectral features (5):** Mean, standard deviation, skewness, kurtosis of the normalized graph Laplacian eigenvalues, and the sum of the 10 smallest non-zero eigenvalues.
3.  **Diffusion map features (15):** Node embeddings derived from the top 5 eigenvectors of the random walk transition matrix, aggregated at the graph level using mean, maximum, and minimum pooling (5 eigenvectors x 3 aggregation methods).

A significant challenge encountered during feature engineering was the computational cost of eigenvalue decomposition for large graphs. A threshold (`MAX_NODES_FOR_EIGS = 500`) was set, above which Laplacian and diffusion feature calculations were skipped, resulting in NaN values for these features. Specifically, 638 out of 800 training graphs and 164 out of 200 test graphs exceeded this size limit. This led to a substantial number of NaN entries in the engineered feature matrices (12,760 NaNs in `X_train_engineered` (an 800x24 matrix) and 3,280 NaNs in `X_test_engineered` (a 200x24 matrix)), primarily affecting the 20 spectral and diffusion features. These NaNs were subsequently handled by mean imputation prior to Principal Component Analysis (PCA).

The distributions of the engineered features (post-imputation for plotting purposes where applicable, using raw values from `data/engineered_features_train.npz`) are visualized in a series of histograms:
*   Edge features (`engineered_feature_dist_edge_1_<timestamp>.png`): These features, such as `mean_delta_sf` and `mean_log_mass_ratio`, show relatively compact distributions.
*   Laplacian spectral features (`engineered_feature_dist_laplacian_2_<timestamp>.png`): Features like `lap_eig_mean` and `lap_eig_std` exhibit varied distributions. The imputation of NaNs for large graphs likely influences these observed distributions.
*   Diffusion map features (mean, max, min pooled) (`engineered_feature_dist_diff_mean_3_<timestamp>.png`, `engineered_feature_dist_diff_max_4_<timestamp>.png`, `engineered_feature_dist_diff_min_5_<timestamp>.png`): Similar to Laplacian features, these distributions are also affected by the imputation strategy for large graphs.

### Dimensionality Reduction of Engineered Features

Principal Component Analysis (PCA) was applied to the mean-imputed 24 engineered features derived from the training set. The goal was to reduce dimensionality while retaining most of the variance.
The PCA explained variance plot (`data/pca_explained_variance_plot_1_<timestamp>.png`, generated in Step 2) shows the cumulative and individual explained variance per component. It was determined that **8 principal components** were sufficient to explain approximately **96.29%** of the variance in the engineered feature space. Consequently, both training and test engineered feature sets were transformed into this 8-dimensional PCA space.

To visualize the PCA-transformed data, scatter plots of the first two principal components (PC1 vs. PC2) were generated, colored by the true values of $\Omega_m$ and $\sigma_8$ (`data/pca_projection_plot_6_<timestamp>.png`).
*   **PC1 vs. PC2 (colored by $\Omega_m$):** This projection does not reveal any immediately obvious strong linear separation or clear clustering of data points corresponding to different $\Omega_m$ values. While some regions might show a higher density of certain $\Omega_m$ values, the overall distribution appears mixed.
*   **PC1 vs. PC2 (colored by $\sigma_8$):** Similarly, the projection colored by $\sigma_8$ does not exhibit distinct visual patterns or separations. The points are largely intermingled, suggesting that the first two principal components of the engineered features may not linearly capture strong cosmological signatures in a visually separable manner.

This lack of clear visual separation in the top two PCs hints that the relationship between the engineered features (even after PCA) and the cosmological parameters might be complex, non-linear, or that these features are not strongly discriminative.

### Baseline Feature Sets

Two baseline approaches were established for comparison:
1.  **Baseline Aggregated Node Features:** For each graph, 16 features were computed by taking the mean, standard deviation, minimum, and maximum of each of the 4 normalized node features across all nodes in that graph. These features provide simple global statistics of the halo properties within each merger tree. No NaNs were present in these baseline features after computation.
2.  **Graph Convolutional Network (GCN):** A GCN model was implemented for graph-level regression. It consisted of two `GCNConv` layers with ReLU activations, followed by a global mean pooling layer and two fully connected layers for regression. The GCN was trained on normalized node features and graph structures for 50 epochs on a CPU.

### Predictive Performance of Cosmological Parameters

Regression models (Random Forest Regressor - RFR, Gradient Boosting Regressor - GBR) were trained on both the PCA-transformed engineered features and the baseline aggregated node features. The GCN provided a deep learning baseline. Performance was evaluated using R-squared (R²) and Mean Squared Error (MSE) on the test set. All hyperparameters for RFR and GBR were tuned using `GridSearchCV` with `GroupKFold` (5 splits) based on `lh_id` to prevent data leakage.

A summary of the model performances is presented in Table 1 and visualized in the performance comparison plots (`data/model_performance_r2_11_<timestamp>.png` for R² and `data/model_performance_mse_12_<timestamp>.png` for MSE).

**Table 1: Model Performance on Test Set for Predicting $\Omega_m$ and $\sigma_8$**

| Feature Set             | Model | Target    | R²      | MSE        |
|-------------------------|-------|-----------|---------|------------|
| **Engineered + PCA**    | RFR   | $\Omega_m$  | -0.2682 | 0.010926   |
|                         | GBR   | $\Omega_m$  | -0.1885 | 0.010240   |
|                         | RFR   | $\sigma_8$  | -0.4622 | 0.016500   |
|                         | GBR   | $\sigma_8$  | -0.4412 | 0.016262   |
| **Baseline Aggregated** | RFR   | $\Omega_m$  | 0.8879  | 0.000966   |
|                         | GBR   | $\Omega_m$  | 0.9134  | 0.000746   |
|                         | RFR   | $\sigma_8$  | 0.2827  | 0.008094   |
|                         | GBR   | $\sigma_8$  | 0.4238  | 0.006502   |
| **GCN**                 | GCN   | $\Omega_m$  | 0.9786  | 0.000185   |
|                         | GCN   | $\sigma_8$  | 0.4977  | 0.005668   |

**Prediction of $\Omega_m$:**
*   The **GCN model** achieved the highest performance, with an R² of **0.9786** and an MSE of **0.000185**. This indicates a very strong predictive capability for $\Omega_m$.
*   The **Baseline Aggregated Node Features** also performed remarkably well. The GBR model yielded an R² of **0.9134** (MSE=0.000746), and the RFR model achieved an R² of **0.8879** (MSE=0.000966).
*   In stark contrast, the **Engineered Features with PCA** performed very poorly. Both RFR (R²=-0.2682) and GBR (R²=-0.1885) resulted in negative R² values, indicating that the models performed worse than a horizontal line (predicting the mean). This suggests these features, in their current form and after PCA, do not capture meaningful information for $\Omega_m$ prediction, or the information is obscured.

**Prediction of $\sigma_8$:**
*   The **GCN model** again showed the best performance for $\sigma_8$, with an R² of **0.4977** and an MSE of **0.005668**. While this is a positive R², it is considerably lower than for $\Omega_m$, suggesting $\sigma_8$ is harder to predict from merger tree morphology.
*   The **Baseline Aggregated Node Features** with GBR achieved an R² of **0.4238** (MSE=0.006502), and with RFR an R² of **0.2827** (MSE=0.008094). These results are modest but significantly better than the engineered features.
*   The **Engineered Features with PCA** again failed to provide predictive power for $\sigma_8$, with RFR (R²=-0.4622) and GBR (R²=-0.4412) yielding negative R² values.

The predicted vs. true value plots further illustrate these findings:
*   For $\Omega_m$ (`data/predicted_vs_true_Omega_m_9_<timestamp>.png`): The GCN plot shows points tightly clustered around the y=x line. The baseline models also show a strong correlation, whereas the PCA-engineered feature models show a scatter with no clear correlation.
*   For $\sigma_8$ (`data/predicted_vs_true_sigma_8_10_<timestamp>.png`): The GCN plot shows a positive correlation, but with more scatter than for $\Omega_m$. The baseline models also exhibit a positive but weaker correlation. The PCA-engineered feature models again show no discernible correlation.

### Feature Importance Analysis

Feature importances were derived for the tree-based models (RFR and GBR).
The plots `data/feature_importances_Omega_m_7_<timestamp>.png` and `data/feature_importances_sigma_8_8_<timestamp>.png` display these importances.

**For Baseline Aggregated Node Features:**
*   **Predicting $\Omega_m$:**
    *   The RFR model highlighted `SF_mean` (mean scale factor of halos in the tree) as highly important, followed by `logM_mean` (mean log mass) and `logV_mean` (mean log Vmax).
    *   The GBR model also emphasized `SF_mean`, `logM_mean`, and `logV_mean`, along with `SF_std` (std of scale factor).
    This suggests that the average formation time (indicated by scale factor) and average mass/potential well depth of halos in a merger tree are strong indicators of $\Omega_m$.
*   **Predicting $\sigma_8$:**
    *   For RFR, `logM_max` (maximum log mass in the tree), `SF_mean`, and `logC_mean` (mean log concentration) were among the top features.
    *   For GBR, `logM_max`, `SF_mean`, and `logM_mean` showed notable importance.
    The importance of maximum mass and concentration metrics for $\sigma_8$ is plausible, as $\sigma_8$ relates to the amplitude of matter fluctuations, influencing the formation of massive structures and their concentrations.

**For Engineered Features + PCA:**
*   The feature importances are for the 8 Principal Components (PC1 to PC8).
*   **Predicting $\Omega_m$ and $\sigma_8$:** For both targets and both RFR/GBR models, various PCs showed some importance, but no single PC consistently dominated across all models and targets. For instance, for $\Omega_m$ with GBR, PC1, PC2, and PC3 showed higher importance. For $\sigma_8$ with GBR, PC1, PC3, and PC5 were more prominent.
*   Interpreting these PC importances directly in terms of the original 24 engineered features is challenging without analyzing the PCA loadings. However, given the overall poor performance of these PCA-reduced features, the specific importances of these PCs are less critical than the observation that they collectively failed to capture predictive signals.

### Discussion of Engineered Graph Features' Performance

The central and surprising outcome of this study is the extremely poor performance of the sophisticated graph spectral and diffusion geometry features, even after PCA. The negative R² values indicate that these features, as implemented and processed, were detrimental to prediction compared to a simple mean prediction. Several factors likely contributed to this:

1.  **Information Loss due to `MAX_NODES_FOR_EIGS` and Imputation:** A large fraction of graphs (638/800 training, 164/200 testing) had their spectral and diffusion features replaced by NaNs because they exceeded the 500-node limit for eigenvalue computation. These NaNs were then mean-imputed. This imputation strategy, applied to a majority of the data for these 20 features, likely homogenized the feature values, masked true structural variations, and introduced noise, severely degrading their informational content. The distributions plotted in `engineered_feature_dist_laplacian_*.png` and `engineered_feature_dist_diff_*.png` are likely dominated by these imputed means for many graphs.
2.  **Insensitivity of Chosen Features or Aggregation:** The specific spectral moments (mean, std, skewness, kurtosis of eigenvalues, sum of smallest eigenvalues) and the aggregation of diffusion map eigenvectors (mean, max, min pooling) might not be the most sensitive probes of cosmological information encoded in merger tree morphology. These global summaries might average out subtle but crucial structural differences.
3.  **Suboptimal PCA Transformation:** While PCA reduces dimensionality by maximizing variance, it is an unsupervised method. The components that explain most of the variance in the feature space are not necessarily the most predictive for the cosmological parameters. It's possible that information relevant to $\Omega_m$ and $\sigma_8$ was present in lower-variance components that were discarded, or that the linear transformation of PCA was insufficient to disentangle complex relationships.
4.  **Coarseness of Edge Features:** The mean and variance of scale factor differences and log mass ratios might be too simplistic to capture the nuances of merger histories.
5.  **Intrinsic Difficulty:** The relationship between these specific graph-theoretic properties and cosmology might be inherently weak or highly non-linear, making it difficult for traditional regressors to capture, even with perfect features.

The failure of these "classical" geometric deep learning inspired features underscores the challenge in manual feature engineering for complex graph data, especially when computational constraints lead to compromises in feature calculation.

### Comparison with Baseline Approaches and Physical Implications

The **Baseline Aggregated Node Features** significantly outperformed the engineered features. For $\Omega_m$, R² values reached up to 0.9134 (GBR), and for $\sigma_8$, up to 0.4238 (GBR). This implies that simple global statistics of halo properties (average mass, concentration, Vmax, and particularly average scale factor, as indicated by feature importances) within a merger tree are strongly correlated with $\Omega_m$ and moderately correlated with $\sigma_8$. Physically, this suggests that cosmologies with different $\Omega_m$ values produce merger trees with distinguishably different average halo properties and formation epochs. For instance, a higher $\Omega_m$ might lead to earlier formation and thus lower average scale factors in trees, which was picked up by the `SF_mean` feature.

The **GCN model** demonstrated the best performance overall, achieving an R² of 0.9786 for $\Omega_m$ and 0.4977 for $\sigma_8$. This highlights the strength of GNNs in learning relevant representations directly from raw node features and graph connectivity. GCNs can automatically discover complex, non-linear patterns and feature interactions that are difficult to hand-engineer. The GCN's success, particularly its improvement over the strong baseline, suggests that not only the global average of node properties but also their specific arrangement and relationships within the tree structure (i.e., the graph topology itself) carry cosmological information. The GCN effectively learns a "graph embedding" that is optimized for the prediction task.

The significantly better performance for $\Omega_m$ compared to $\sigma_8$ across all successful methods (Baseline and GCN) suggests that merger tree morphology, as characterized by the input node features, is more sensitive to changes in the matter density parameter than to the amplitude of matter fluctuations on 8 $h^{-1}$Mpc scales.

### Limitations

This study has several limitations:
1.  **Engineered Feature Calculation:** The `MAX_NODES_FOR_EIGS` threshold and subsequent massive imputation for spectral and diffusion features is the most significant limitation, likely rendering these features ineffective.
2.  **Scope of Engineered Features:** The selection of spectral and diffusion features was not exhaustive. Other graph invariants or embedding techniques might yield better results.
3.  **Computational Resources:** GCN training was limited by CPU availability and a modest number of epochs (50). More extensive training or GPU acceleration could potentially improve GCN performance further.
4.  **Hyperparameter Optimization:** While performed, the search grids for RFR/GBR were kept relatively small for computational efficiency.
5.  **Interpretability of GCN:** While GCNs perform well, their learned representations are often black boxes, making direct physical interpretation challenging without specialized explainability techniques.
6.  **Dataset Size:** While 1000 trees provide a reasonable starting point, larger datasets from more diverse simulations could improve model robustness and allow for finer distinctions.

### Future Work

Based on these findings, several avenues for future research emerge:
1.  **Addressing Computational Bottlenecks for Engineered Features:** Explore methods for approximating spectral or diffusion properties for large graphs or use graph coarsening techniques. Alternatively, focus on features that are computationally cheaper yet structurally informative.
2.  **Advanced Graph Representation Learning:** Investigate more sophisticated GNN architectures (e.g., Graph Attention Networks, GraphSAGE) and different pooling strategies. Training for more epochs with more extensive hyperparameter tuning could also be beneficial.
3.  **Explainable AI for GNNs:** Apply GNN explainability methods (e.g., GNNExplainer, Integrated Gradients) to understand which nodes, edges, or sub-structures in the merger trees are most influential for predicting cosmological parameters. This could yield new physical insights.
4.  **Hybrid Models:** Combine learned GNN embeddings with astrophysically motivated engineered features (perhaps more refined versions than those used here) or the successful baseline aggregated features.
5.  **Refined Edge and Path Features:** Explore more detailed characterizations of merger events, such as mass ratios over time, merger timings relative to main branch formation, or features based on paths within the tree.
6.  **Alternative Dimensionality Reduction:** If pursuing engineered features, explore supervised dimensionality reduction techniques instead of PCA, which might better preserve task-relevant information.
7.  **Sensitivity Analysis:** Investigate the sensitivity of predictions to different aspects of the merger tree construction algorithm or simulation resolution.

In conclusion, while the initially proposed "classical" graph spectral and diffusion geometry features proved ineffective under the current implementation and computational constraints, the study successfully demonstrates that simpler aggregated node features hold significant predictive power for $\Omega_m$ and moderate power for $\sigma_8$. Furthermore, a basic GCN model substantially outperforms these methods, underscoring the potential of graph neural networks to automatically learn cosmologically relevant information from the complex structure of dark matter halo merger trees. The challenge remains in either refining classical graph features to be both informative and computationally tractable or in leveraging GNNs more deeply while enhancing their interpretability.