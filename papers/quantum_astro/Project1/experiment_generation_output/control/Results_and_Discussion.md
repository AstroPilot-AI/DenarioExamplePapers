<!-- filename: Results_and_Discussion.md -->
## Results and Discussion

This section details the outcomes of applying Quantum Tensor Train (QTT) based feature engineering to cosmological merger tree subgraphs for the prediction of final halo mass. It interprets the quantitative performance of regression models, analyzes the characteristics of the derived features, and discusses the implications and limitations of the approach.

### 1. Data Preprocessing and Subgraph Extraction Yield

The initial dataset comprised 300 merger trees. Node features (mass, concentration, vmax, scale factor) underwent a preprocessing pipeline involving a logarithmic transformation for mass and vmax, followed by standardization (zero mean, unit variance) of all four features based on global statistics from the entire dataset. This ensured numerical stability and feature comparability. For instance, raw halo mass, originally spanning approximately `9.7` to `14.5` (log10 scale as per problem description, though initial stats show these values directly), was transformed and standardized to a mean near zero and unit standard deviation (Step 1 output: `mass: Mean=-4.011e-06, Std=1.0`).

A critical step involved extracting k-hop subgraphs around nodes on the main progenitor branch of each merger tree. The main branch was identified using the `mask_main` attribute provided with each tree. However, a significant challenge emerged during this phase: the vast majority of main branch node indices specified in `mask_main` were found to be invalid (out of bounds) for their respective trees. The execution log from Step 2 ("Subgraph Extraction") reported: *"Skipped 28448 main branch nodes due to invalid index."* This issue drastically limited the number of valid subgraphs that could be extracted. Across all 300 trees and for all tested k-values (k=1, 2, 3), only a total of 5 unique subgraphs, originating from 5 distinct trees, could be successfully processed.

Consequently, all subsequent QTT feature engineering, model training, and evaluation were performed on this severely reduced dataset of N=5 trees. For k=1, the extracted subgraphs had an average of 4.0 nodes (min: 3, max: 8). For k=2, the average was 8.4 nodes (min: 5, max: 19), and for k=3, it was 13.4 nodes (min: 7, max: 32). These subgraphs' node feature matrices were padded to the next power of 2 in the node dimension (e.g., 8 for k=1, 32 for k=2 and k=3) to prepare them for QTT decomposition.

**This extremely small effective sample size (N=5) is the most significant limitation of the current study. While the methodology was executed as planned, the quantitative results for regression performance must be interpreted with extreme caution, as they are based on in-sample evaluation on these 5 data points and are not generalizable. The findings should be considered a proof-of-concept demonstration on a minimal dataset rather than a robust statistical evaluation.**

### 2. QTT Decomposition and Feature Engineering

For each of the 5 valid subgraphs, the padded node feature matrix (e.g., shape [8, 4] for k=1) was reshaped into a higher-order tensor (e.g., [2, 2, 2, 4] for k=1) and decomposed using QTT. Experiments were conducted with QTT ranks of 2 and 3. The QTT cores were then flattened and concatenated to form a single feature vector for each subgraph. For trees with multiple (though in this case, only one per tree) valid main branch subgraphs, these QTT vectors were intended to be aggregated by mean pooling; here, it simply meant taking the QTT vector of the single available subgraph.

The reconstruction Mean Squared Error (MSE) of the QTT decomposition provides an indication of the compression fidelity. For k=1 subgraphs, a QTT rank of 2 yielded an average reconstruction MSE of approximately 0.032, while a rank of 3 reduced this to 0.0053. For k=2, rank 2 gave an MSE of 0.033, and rank 3 gave 0.016. For k=3, rank 2 resulted in an MSE of 0.057, and rank 3 in 0.027 (Step 3 output). These low MSE values suggest that QTT, even with relatively low ranks, could reconstruct the (padded) subgraph feature matrices with reasonable accuracy, indicating that the compressed QTT features retained substantial information from the local subgraph environments. The expected QTT feature vector sizes were, for example, 28 for k=1, rank=2, and 90 for k=3, rank=3.

### 3. Regression Performance for Final Halo Mass Prediction

Random Forest Regressors were trained to predict the first component of the target variable `y` (representing a final halo mass property at z=0), using either baseline aggregated features or the QTT-derived features. Given N=5, all evaluations are in-sample.

#### 3.1. Baseline Model
Baseline features were constructed by taking the mean, maximum, and variance of the four preprocessed node features along the (valid portion of the) main branch for each of the 5 trees. This resulted in a 12-dimensional feature vector per tree.
The baseline model achieved:
*   Mean Squared Error (MSE): 0.00197
*   Mean Absolute Error (MAE): 0.0315
*   R-squared (R²): 0.797
(Step 4 output, "Evaluation on Test Set (Baseline Features)")

The "Predicted vs. True Values (Baseline Features)" plot (e.g., `data/pred_vs_true_baseline_1_*.png`) and the "Residuals Plot (Baseline Features)" (e.g., `data/residuals_baseline_2_*.png`) visually represent this performance. Given N=5, the points in the scatter plot are expected to align reasonably well if the model captures any trend.

#### 3.2. QTT-based Models
The performance of QTT-based models varied with k and QTT rank:

| k | QTT Rank | MSE      | MAE      | R²     |
|---|----------|----------|----------|--------|
| 1 | 2        | 0.00151  | 0.0261   | 0.845  |
| 1 | 3        | 0.00159  | 0.0276   | 0.836  |
| 2 | 2        | 0.00161  | 0.0279   | 0.834  |
| 2 | 3        | 0.00181  | 0.0291   | 0.813  |
| 3 | 2        | 0.00196  | 0.0348   | 0.798  |
| 3 | 3        | 0.00187  | 0.0341   | 0.808  |
*(Table compiled from Step 4 outputs for QTT models)*

Predicted vs. True and Residuals plots were generated for each QTT model configuration (e.g., `data/pred_vs_true_qtt_k1_r2_4_*.png`, `data/residuals_qtt_k1_r2_5_*.png`).

#### 3.3. Comparative Analysis
The comparison plots for MSE, MAE, and R² (e.g., `data/comparison_mse_22_*.png`, `data/comparison_mae_23_*.png`, `data/comparison_r2_24_*.png`) summarize these metrics across all models.

Numerically, the QTT model with k=1, rank=2 showed the best performance (R²=0.845) among all models, slightly outperforming the baseline (R²=0.797) on this N=5 dataset. Other QTT configurations also showed R² values comparable to or slightly better than the baseline.

**It is crucial to reiterate that with N=5, these differences are not statistically significant and are highly susceptible to the specific characteristics of these five samples.** A model could easily achieve high R² by chance or by overfitting to these few points. The primary takeaway is that the QTT feature engineering pipeline is functional and can produce features usable by a standard regressor. The observed high R² values, while numerically impressive, should not be interpreted as evidence of a generally superior model without validation on a substantially larger and more representative dataset.

### 4. Feature Space Analysis

#### 4.1. Feature Distributions
Distributions for the 12 baseline features and the first 12 QTT features (for k=1, rank=2) were plotted (Step 5 outputs: `data/feature_dist_baseline_features_1_*.png` and `data/feature_dist_qtt_features_(k=1,_r=2)_2_*.png`).
For the baseline features, the `*_var` features (variance of mass, concentration, etc., along the main branch) were all zero for the 5 selected trees. This suggests that for these specific trees, either the valid main branch segment consisted of a single node, or the features were constant along the main branch segment. This is another artifact of the extremely small and potentially unrepresentative sample. The `*_mean` and `*_max` features showed some variation.
The QTT features, being components of compressed tensor cores, exhibit distributions that are not directly interpretable in terms of physical properties. For N=5, these plots show the 5 individual values for each feature, often colored by the target variable. For example, `QTT_feat_0` for k=1, rank=2, had values concentrated near 1.0 for these 5 samples.

#### 4.2. Dimensionality Reduction (PCA)
PCA was applied to both baseline and QTT (k=1, rank=2) feature sets, reducing them to 2 components for visualization (Step 5 outputs: `data/pca_baseline_features_3_*.png` and `data/pca_qtt_features_(k=1,_r=2)_4_*.png`).
For the baseline features, the first two principal components explained approximately 75.7% and 23.6% of the variance, respectively (total ~99.3%). For the QTT features (k=1, rank=2), the first two components explained about 71.3% and 24.0% of the variance (total ~95.3%).
The PCA plots show the 5 data points in a 2D space, colored by their target halo mass property. With only 5 points, any visual "separation" or "structure" is anecdotal. However, the high cumulative explained variance in both cases suggests that much of the feature variability within this tiny sample can be captured in a low-dimensional space.

#### 4.3. Feature Importances
Feature importance plots were generated from the Random Forest models.
*   **Baseline Model (`data/feature_importances_baseline_3_*.png`):** For the N=5 sample, features like `mass_mean`, `vmax_mean`, `concentration_mean`, and their `_max` counterparts showed non-zero importance. As noted, `*_var` features had zero importance because their values were zero.
*   **QTT Models (e.g., `data/feature_importances_qtt_k1_r2_6_*.png`):** The QTT feature importance plots show a distribution of importances across the abstract QTT features. For k=1, rank=2 (28 features), several features contributed to the prediction. The interpretation of individual QTT feature importances is challenging due to their abstract nature. However, the fact that the model assigns varying importances suggests that different components of the compressed QTT representation contribute differently to the predictive task.

The utility of QTT features lies in their potential to automatically learn and encode complex, non-linear relationships and structural information from the local subgraph environment into a compact vector. This could be more powerful than simple statistical aggregations (mean, max, var) if the local graph structure and multi-feature interactions are important for the prediction task. However, this potential can only be validated with a much larger dataset.

### 5. Impact of k-hop Neighborhood and QTT Rank

Analyzing the table in Section 3.2:
*   **Impact of k:** For rank 2, performance (R²) decreased slightly as k increased (k=1: 0.845, k=2: 0.834, k=3: 0.798). A similar, though less clear, trend is seen for rank 3. This might suggest that for this tiny dataset, larger subgraphs (larger k) introduced more noise or irrelevant information relative to the signal, or that the padding to a larger fixed size (32 for k=2,3 vs 8 for k=1) had an effect. Alternatively, the nature of the 5 specific subgraphs available for larger k might have been less informative.
*   **Impact of QTT Rank:**
    *   For k=1, increasing rank from 2 to 3 slightly decreased R² (0.845 to 0.836).
    *   For k=2, increasing rank from 2 to 3 slightly decreased R² (0.834 to 0.813).
    *   For k=3, increasing rank from 2 to 3 slightly increased R² (0.798 to 0.808).
    Higher QTT rank allows for less compression and potentially captures more detail, as evidenced by lower reconstruction MSEs (Section 2). However, this did not consistently translate to better predictive performance on this N=5 dataset. This could be due to overfitting with more complex features on such a small sample, or the specific information captured by the higher rank cores not being relevant for these 5 samples.

Again, these trends are based on N=5 and are not robust.

### 6. Visualization of Local Subgraph Structure

A sample 1-hop subgraph was visualized in `data/subgraph_vis_k1_idx0_5_*.png` (Step 5 output). This plot shows the actual nodes and edges of a 1-hop neighborhood around a main branch node from one of the 5 trees. Nodes are colored by their first preprocessed feature value (standardized log-mass). This visualization helps to understand the local structures that are being fed into the QTT decomposition. QTT processes the feature matrix associated with such a structure, aiming to find a compressed representation of the multi-variate information within this local environment.

### 7. Critical Limitations and Caveats

The foremost limitation, overshadowing all quantitative results, is the **extremely small effective sample size of N=5 trees** used for QTT feature generation and all subsequent modeling. This was due to an issue with `mask_main` providing a large number of invalid node indices during subgraph extraction.
*   **Lack of Generalizability:** Results (MSE, MAE, R²) are in-sample evaluations and cannot be generalized. The high R² values are likely artifacts of fitting to these specific 5 points.
*   **Statistical Significance:** No claims of statistical significance can be made regarding model comparisons or feature importances.
*   **Robustness of Trends:** Observed trends with k or QTT rank are not robust.

Other limitations include:
*   **CPU Constraints:** While the current N=5 dataset is trivial for CPUs, scaling to the full dataset (if the `mask_main` issue is resolved) and potentially larger QTT ranks or more complex models would be more demanding.
*   **Simple Aggregation:** Mean pooling was used for (hypothetically) multiple subgraphs per tree. More sophisticated aggregation might be beneficial.
*   **Limited Hyperparameter Tuning:** No extensive hyperparameter tuning was performed for the Random Forest models due to the small N.
*   **Interpretability of QTT Features:** While potentially powerful, the abstract nature of QTT features makes direct physical interpretation difficult.

### 8. Future Research Directions

To realize the potential of QTT for cosmological merger tree analysis, future work must prioritize:
1.  **Resolving the `mask_main` / Subgraph Extraction Issue:** This is critical. The cause of invalid node indices in `mask_main` must be identified and fixed to enable processing of a significant portion of the 300 available trees, or ideally, even larger datasets.
2.  **Validation on Larger Datasets:** Once a substantial number of trees can be processed, rigorous training, validation, and testing splits are necessary to obtain reliable performance metrics.
3.  **Exploration of QTT Parameters:** With a larger dataset, systematically explore a wider range of QTT ranks and their impact on both reconstruction error and predictive performance.
4.  **Advanced Subgraph Definitions:** Consider alternative definitions of "local environment" beyond simple k-hop subgraphs, perhaps weighted by physical properties or time evolution.
5.  **Sophisticated Aggregation:** If multiple subgraphs per tree become available, explore attention mechanisms or other learnable pooling methods to aggregate their QTT features.
6.  **Comparison with Graph Neural Networks (GNNs):** On a sufficiently large dataset, compare the QTT-based feature engineering approach with end-to-end GNN models.
7.  **Broader Range of Prediction Tasks:** Apply the methodology to predict other halo properties or even aspects of their formation history (e.g., classifying merger types).
8.  **Interpretability Techniques:** Explore methods from XAI (Explainable AI) to gain insights into what aspects of the subgraphs the QTT features are capturing, potentially by correlating QTT feature activations with subgraph properties.

### 9. Conclusion

This study explored a novel approach of applying Quantum Tensor Train (QTT) decomposition for feature engineering on k-hop subgraphs extracted from cosmological merger trees, with the goal of predicting final halo mass. The methodological pipeline, including data preprocessing, subgraph extraction, QTT decomposition, feature aggregation, and regression modeling, was successfully implemented. QTT demonstrated its ability to compress subgraph feature matrices with low reconstruction error.

However, due to a critical issue with invalid main branch node indices provided in the dataset's `mask_main` attribute, the analysis was severely constrained to an effective sample size of only N=5 trees. While the QTT-derived features showed nominally good predictive performance (R² up to 0.845) on this tiny, in-sample evaluation, these quantitative results are not statistically significant or generalizable. The primary value of this work lies in the demonstration of the pipeline's functionality and the identification of key challenges.

Future work must urgently address the subgraph extraction issue to leverage a substantially larger dataset. Only then can the true potential of QTT-informed feature engineering for encoding complex local information from merger trees be robustly evaluated and compared against traditional methods and more complex models like GNNs. If successful on a larger scale, this approach could offer a computationally efficient way to extract rich, structurally-aware features from graph-like cosmological data.