<!-- filename: results_and_discussion.md -->
## Results and Discussion

This study aimed to investigate the relationship between dark matter halo merger tree morphology and halo assembly bias, employing Graph Neural Networks (GNNs) guided by insights from Topological Data Analysis (TDA). The GNN was intended to predict an assembly bias proxy from the structural and physical properties of halo merger trees. However, significant challenges encountered during the data preprocessing phase, primarily concerning the calculation of the assembly bias proxy, severely limited the scope and reliability of the subsequent GNN analysis. The TDA component, planned to extract topological features and inform the GNN, was not executed as per the project's progression.

### 1. Data Preprocessing and Dataset Characteristics

The initial dataset comprised 1000 merger trees from cosmological N-body simulations. Each tree's nodes represent dark matter halos, characterized by four features: halo mass, halo concentration, maximum circular velocity (Vmax), and scale factor (ranging from 0 to 1). The edges in the tree represent merger events.

**1.1. Feature Engineering and Normalization:**
Node features underwent preprocessing as planned: halo mass and Vmax were log10-transformed to manage their typically large dynamic ranges and skewed distributions. All four node features (log10(Mass), Concentration, log10(Vmax), Scale Factor) were then normalized to a [0, 1] range across the dataset. The normalization parameters, derived from the subset of data for which the assembly bias proxy could be computed, were:
*   **Log10(Mass):** Min = 0.983, Max = 1.167
*   **Concentration:** Min = 0.0002, Max = 3.767
*   **Log10(Vmax):** Min = 0.194, Max = 0.499
*   **Scale Factor:** Min = 0.077, Max = 1.0

Edge features were engineered to capture information about the merger events. These included:
1.  **Mass Ratio:** The ratio of the smaller halo mass to the larger halo mass involved in a merger.
2.  **Time Difference:** The absolute difference in scale factors between the merging halos.
These engineered edge features were also normalized to a [0, 1] range:
*   **Mass Ratio:** Min = 0.714, Max = 1.0
*   **Time Difference:** Min = 0.003, Max = 0.776

**1.2. Assembly Bias Proxy Calculation and Dataset Reduction:**
A critical step was the calculation of an assembly bias proxy for each merger tree. This proxy was defined as the mean halo mass of the main progenitor halos existing at redshift z=0 (scale factor ≈ 1), identified using the `mask_main` attribute provided in the raw data.

This stage encountered significant difficulties. As reported in Step 1 of the execution summary: "Warning: 975 trees had issues with assembly bias proxy calculation (set to NaN initially, may have used fallback, or mask_main was problematic)." Consequently, **975 out of the 1000 available merger trees (97.5%) were discarded** because a valid assembly bias proxy could not be computed.

This drastic reduction left an operational dataset of only **25 merger trees**. This extremely small sample size became the single most significant limiting factor for the entire study. The 25 trees were split into:
*   Training set: 17 samples
*   Validation set: 3 samples
*   Test set: 5 samples

The statistics for the assembly bias proxy (log10(Mass) at z=0) for these 25 trees were:
*   Mean: 11.22
*   Standard Deviation: 0.857
*   Min: 10.55
*   Max: 13.07

The variance of the assembly bias proxy in this small dataset is approximately (0.857)^2 ≈ 0.734. This value serves as a crucial benchmark for interpreting the Mean Squared Error (MSE) of the GNN model.

The inability to compute the assembly bias proxy for the vast majority of trees suggests potential issues with the `mask_main` field's reliability in identifying the main branch, the definition of "z=0" halos (e.g., the scale factor threshold of ≥0.99), or inconsistencies in the raw data structure itself. Without resolving this, any conclusions drawn from models trained on such a small and potentially unrepresentative subset of data are highly speculative.

### 2. Topological Data Analysis (TDA)

The original research plan included a Topological Data Analysis (TDA) component. This involved:
1.  Converting each merger tree into a simplicial complex, using the scale factor as a filtration parameter.
2.  Computing persistent homology (H0 for connected components, H1 for loops) to generate persistence diagrams.
3.  Extracting topological features from these diagrams (e.g., Betti numbers, persistence of features).
4.  Analyzing the correlation between these topological features and the assembly bias proxy.

The intention was that these TDA-derived features could provide quantitative measures of merger tree morphology, potentially offering insights into halo formation history that correlate with assembly bias. These insights could then guide the GNN architecture or serve as additional input features.

However, **Step 2 (Topological Data Analysis) was not executed** during the project's progression. Consequently, no topological features were extracted, and no correlation analysis between such features and the assembly bias proxy was performed. This omission means that one of the key innovative aspects of the proposed methodology—the synergy between TDA and GNNs—could not be explored. The GNN development proceeded without any guidance or input from TDA.

### 3. Graph Neural Network (GNN) Model Performance

A Graph Convolutional Network (GCN) was designed and implemented using PyTorch Geometric to predict the assembly bias proxy from the preprocessed merger tree data.

**3.1. GNN Architecture and Training:**
The GCNNet architecture included:
*   A node feature encoder (Linear layer).
*   An edge feature encoder (Linear layer), if edge features were present.
*   A configurable number of `GCNConv` layers with ReLU activation.
*   A global mean pooling layer to aggregate node embeddings into a graph-level embedding.
*   A final linear regression layer to output the scalar assembly bias proxy.

Hyperparameter tuning was attempted for the number of GCN layers ([1, 2]) and the number of hidden channels ([16, 32]). Other training parameters were fixed:
*   Learning Rate: 0.0001
*   Batch Size: 4
*   Number of Epochs: 50
*   Optimizer: Adam
*   Loss Function: Mean Squared Error (MSE)
*   Weight Decay: 1e-5

**3.2. Hyperparameter Tuning and Training Instability:**
The hyperparameter tuning phase revealed significant issues, particularly with smaller network configurations.
*   Models with `hidden_channels: 16` (for both 1 and 2 GCN layers) consistently produced `NaN` (Not a Number) losses during training and validation. This indicates severe training instability, possibly due to vanishing/exploding gradients, issues with the very small batch sizes relative to model complexity (even if small), or numerical precision problems exacerbated by the extremely limited data. Gradient clipping was implemented, but it was not sufficient to prevent NaNs in these cases.

The configurations with `hidden_channels: 32` were trainable:
*   **1 GCN layer, 32 hidden channels:** Achieved the best validation MSE of **104.7496**.
*   **2 GCN layers, 32 hidden channels:** Achieved a validation MSE of **105.6353**.

The training and validation loss curves for the best configuration (1 GCN layer, 32 hidden channels) are presented in the plot `data/training_validation_loss_curves_plot_1_1748137938.png`. While both training and validation losses show a decreasing trend over the 50 epochs, the final MSE values are extraordinarily high. For context, the variance of the target variable (assembly bias proxy) in the dataset is approximately 0.734. An MSE exceeding 100 is orders of magnitude larger, suggesting the model has learned very little, if anything, about the underlying relationship between the input features and the target.

**3.3. Test Set Evaluation:**
The GNN model with the best hyperparameters (1 GCN layer, 32 hidden channels, validation MSE ≈ 104.75) was evaluated on the test set, which comprised only 5 samples. The performance metrics were:
*   **Mean Squared Error (MSE): 107.6264**
*   **R-squared (R²): -480.5708**

**Interpretation of Test Metrics:**
*   **MSE:** An MSE of 107.63 on the test set is consistent with the high validation MSE. It confirms that the model's poor performance generalizes to unseen data (albeit a tiny amount). This value is drastically higher than the variance of the target variable (0.734), indicating that the model's predictions are, on average, very far from the true values.

*   **R-squared (R²):** The R² score, or coefficient of determination, measures the proportion of the variance in the dependent variable that is predictable from the independent variables. An R² of 1 indicates perfect prediction. An R² of 0 indicates that the model performs no better than a naive model that always predicts the mean of the target variable. A negative R² value, such as the **-480.5708** observed here, signifies an exceptionally poor fit. It means that the model's predictions are substantially worse than simply predicting the mean of the assembly bias proxy for all test samples. The sum of squared residuals (errors) from the model is vastly larger than the total sum of squares (variance of the data).

The plot `data/predicted_vs_true_bias_plot_2_1748137938.png` visually corroborates these metrics. This scatter plot of predicted versus true assembly bias proxy values on the test set shows points scattered far from the ideal y=x line, indicating a lack of correlation between predictions and actual values. Given the extremely small number of test points (5), this plot should be interpreted with caution, but it aligns with the quantitative R² and MSE.

### 4. Overall Discussion and Significance of Results

The primary outcome of this investigation is that the GNN model, under the severe constraint of an extremely limited dataset (N=25), failed to demonstrate any meaningful predictive capability for the assembly bias proxy based on merger tree morphology. The quantitative metrics (high MSE, large negative R²) unequivocally point to a model that has not learned the underlying patterns in the data.

The significance of these results is largely a cautionary tale about the prerequisites for applying complex machine learning models like GNNs. While GNNs are powerful tools for graph-structured data, their effectiveness is contingent upon sufficient data for training. With only 17 training samples, 3 validation samples, and 5 test samples, the model is starved of information, making it impossible to learn robust representations or generalize effectively. Overfitting is a major risk, but in this case, the model did not even achieve low training error, suggesting it struggled to fit even the tiny training set meaningfully, possibly due to the inherent noise or complexity relative to the signal in such a small sample.

The NaN losses encountered with smaller hidden dimensions further highlight the instability of training deep learning models on insufficient data. The choice of learning rate, batch size, and architecture becomes extremely sensitive.

### 5. Limitations of the Study

This study faced several critical limitations:

1.  **Extreme Dataset Size:** This is the foremost limitation. The failure to compute the assembly bias proxy for 97.5% of the original dataset rendered the subsequent GNN analysis statistically unsound. The results obtained from 25 trees cannot be reliably generalized.
2.  **Assembly Bias Proxy Calculation:** The underlying cause for the failure in calculating the assembly bias proxy for most trees was not resolved. This could stem from issues in the raw data (e.g., `mask_main` field, halo properties at z=0), the definition of the proxy, or the implementation of its calculation. This step requires thorough debugging and validation.
3.  **Skipped Topological Data Analysis (TDA):** The TDA component was not performed. This means that the potential benefits of using topological features to quantify merger tree morphology, understand their correlation with assembly bias, or guide GNN design were not realized.
4.  **Model Complexity and Hyperparameter Tuning:** While a small grid search was performed, the extremely small dataset makes it difficult to ascertain truly optimal hyperparameters or to determine if the chosen GCN architecture was appropriate. The model might still be too complex for the available data, even with few layers and hidden units.
5.  **Lack of Baseline Models:** Due to the focus on the GNN and the data issues, simpler baseline models (e.g., a model based on only the final halo's properties, or a simple regression on a few hand-picked graph features) were not implemented for comparison.
6.  **Generalizability:** Given the data originates from a specific set of simulations ("Pablo_merger_trees2.pt"), the generalizability of any findings (even if positive) to other simulations or observational data would require further investigation.

### 6. Implications for Cosmological Structure Formation Studies and Future Work

Due to the aforementioned limitations and the negative predictive performance, the current study does not provide new astrophysical insights into the connection between merger tree morphology and assembly bias. The hypothesis that GNNs can predict assembly bias from merger trees remains largely untested by this specific endeavor.

However, the proposed methodology is, in principle, sound and warrants further investigation should the data-related challenges be overcome. The following directions for future work are recommended:

1.  **Resolve Assembly Bias Proxy Calculation:** This is paramount. The reasons for the failure to compute the proxy for 97.5% of the trees must be identified and addressed. This may involve:
    *   Debugging the script that uses `mask_main` and `node_halo_id`.
    *   Verifying the consistency and correctness of the `mask_main` field in the raw data.
    *   Re-evaluating the criteria for identifying z=0 halos (e.g., scale factor threshold, completeness of main branch).
    *   Exploring alternative definitions or calculations of an assembly bias proxy if the current one proves intractable with the given data.

2.  **Utilize Full Dataset:** Once a reliable assembly bias proxy can be computed for a significant majority of the 1000 trees (or an even larger dataset), the GNN experiments should be re-conducted.

3.  **Implement Topological Data Analysis:** The TDA component should be implemented as originally planned. Extracted topological features can be:
    *   Correlated with the assembly bias proxy to identify promising morphological indicators.
    *   Used as additional input features to the GNN.
    *   Used to inform the GNN architecture (e.g., by highlighting relevant scales or structures).

4.  **Expand Hyperparameter Tuning and Model Exploration:** With a larger dataset, a more extensive hyperparameter search (learning rate, batch size, number of layers, hidden dimensions, dropout rates, types of GNN layers) can be performed. Different GNN architectures (e.g., Graph Attention Networks (GATs), GraphSAGE) could also be explored.

5.  **Investigate Training Instability:** The causes of NaN losses should be thoroughly investigated, especially if they persist with more data. This might involve examining gradient norms, activation statistics, and the impact of different normalization schemes or initializations.

6.  **Establish Baseline Models:** Compare GNN performance against simpler machine learning models or physics-informed analytical approximations to quantify the added value of the GNN approach.

7.  **Cross-Validation:** Employ robust cross-validation techniques with a larger dataset to obtain more reliable estimates of model performance and generalization.

### 7. Potential Astrophysical Implications (If Successful in the Future)

If the proposed approach, combining GNNs and TDA, were successfully implemented on a sufficiently large and well-processed dataset, it could have several astrophysical implications:

*   **Understanding Assembly Bias:** It could provide a novel, data-driven framework to quantify how the intricate formation history of dark matter halos, as encoded in their merger trees, contributes to assembly bias. This is a key unsolved problem in cosmology.
*   **Improving Galaxy-Halo Connection Models:** Assembly bias affects galaxy clustering. A better understanding and prediction of assembly bias can lead to more accurate models of how galaxies populate dark matter halos.
*   **Informing Semi-Analytic Models (SAMs):** SAMs of galaxy formation rely on merger trees. Insights from GNN/TDA analysis could help refine the prescriptions used in SAMs regarding how halo history influences galaxy properties.
*   **Interpreting Observational Data:** Predictive models for assembly bias could help in interpreting large-scale structure surveys and galaxy clustering measurements, potentially allowing for tighter constraints on cosmological parameters by disentangling baryonic physics from underlying dark matter distributions.
*   **Guidance for Future Simulations:** Identifying which aspects of merger tree morphology are most predictive of assembly bias could guide the design and analysis of future N-body simulations, focusing computational resources on capturing the most relevant physical processes.

In conclusion, while this specific project iteration was severely hampered by data preprocessing challenges, leading to inconclusive GNN performance, the overarching goal of using geometric deep learning and topological data analysis to understand halo assembly bias from merger trees remains a compelling avenue for future research. Addressing the identified data issues is the critical next step to unlock the potential of this approach.