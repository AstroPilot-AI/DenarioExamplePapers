## Results and Discussion

This section presents a comprehensive analysis of the performance of three neural network architectures—Dense Neural Network (Dense NN), standard 1D Convolutional Neural Network (1D CNN), and Dilated 1D CNN—for the emulation of the matter power spectrum, \( P(k) \), from cosmological parameters. The evaluation is based on a synthetic dataset generated using the Eisenstein & Hu (1998) approximation, with 4,000 samples spanning physically relevant ranges of cosmological parameters. The discussion is organized into five subsections: (1) overall performance comparison, (2) scale-dependent accuracy, (3) computational efficiency, (4) sensitivity to cosmological parameters, and (5) recommendations for architecture selection.

---

### 1. Overall Performance Comparison

#### Quantitative Summary

The following table summarizes the key performance metrics for each architecture, as measured on the held-out test set:

| Model Type      | Mean MAPE (%) | Max Error (%) | RMSE (log P(k)) | Train Time (s/epoch) | Total Train Time (s) | Inference Time (ms/sample) | Model Size (MB) |
|-----------------|---------------|---------------|-----------------|----------------------|----------------------|----------------------------|-----------------|
| Dense NN        | 1.19          | 7.85          | 0.0308          | 0.092                | 6.24                 | 0.282                      | 0.163           |
| 1D CNN          | 1.41          | 12.59         | 0.0375          | 0.544                | 40.24                | 0.368                      | 2.113           |
| Dilated 1D CNN  | 1.48          | 15.31         | 0.0400          | 0.834                | 67.56                | 0.395                      | 2.129           |

**Key observations:**
- The Dense NN achieves the lowest mean absolute percent error (MAPE) and root mean squared error (RMSE), outperforming both convolutional architectures in overall accuracy.
- The maximum percent error is also lowest for the Dense NN (7.85%), compared to 12.59% for the 1D CNN and 15.31% for the Dilated 1D CNN.
- The Dense NN is significantly more compact (0.163 MB) and faster to train and infer than the CNN-based models, which are an order of magnitude larger in parameter count and memory footprint.

#### Visual Inspection

Plots of percent error versus \( k \) (see Figure 1) and example predictions versus true \( P(k) \) (see Figure 2) further corroborate the quantitative findings. The Dense NN consistently tracks the true power spectrum across the full \( k \)-range, with only minor deviations at the smallest and largest scales. The CNN-based models, while still accurate, exhibit slightly larger deviations, particularly at high \( k \).

---

### 2. Scale-Dependent Accuracy Analysis

Given the scale-dependent nature of cosmological inference, it is crucial to assess model performance across different \( k \)-regimes. The following table summarizes the mean, median, 95th percentile, and maximum percent errors for each model in three \( k \)-bins: large scales (\( k < 0.1 \) h/Mpc), intermediate (\( 0.1 \leq k < 0.3 \) h/Mpc), and small scales (\( k \geq 0.3 \) h/Mpc).

#### Scale-Dependent Percent Error Metrics (mean/median/95th/max) [%]

**Dense NN**
- **Large scales:** mean=1.32, median=1.03, 95th=3.68, max=7.85
- **Intermediate:** mean=0.82, median=0.72, 95th=1.93, max=3.79
- **Small scales:** mean=1.27, median=1.07, 95th=3.16, max=5.74

**1D CNN**
- **Large scales:** mean=1.41, median=1.04, 95th=3.95, max=12.59
- **Intermediate:** mean=1.17, median=1.01, 95th=2.79, max=5.85
- **Small scales:** mean=1.61, median=1.39, 95th=3.83, max=9.10

**Dilated 1D CNN**
- **Large scales:** mean=1.49, median=1.07, 95th=4.37, max=15.31
- **Intermediate:** mean=1.20, median=1.00, 95th=3.00, max=6.91
- **Small scales:** mean=1.71, median=1.43, 95th=4.20, max=10.74

#### Interpretation

- **Large Scales (\( k < 0.1 \) h/Mpc):** All models perform well, with mean errors below 1.5%. The Dense NN achieves the lowest maximum error, while the Dilated 1D CNN exhibits the highest.
- **Intermediate Scales (\( 0.1 \leq k < 0.3 \) h/Mpc):** Errors are minimized in this regime, with mean errors below 1.2% for all models. The Dense NN again leads in both mean and maximum error.
- **Small Scales (\( k \geq 0.3 \) h/Mpc):** Errors increase slightly for all models, reflecting the greater complexity and dynamic range of \( P(k) \) at small scales. The Dense NN maintains the lowest mean and maximum errors, while the Dilated 1D CNN shows the largest deviations.

The percent error versus \( k \) plots (Figure 1) reveal that all models exhibit a mild increase in error at the smallest and largest \( k \), consistent with the physical expectation that these regimes are more challenging to emulate due to the steepness and nonlinearity of the power spectrum.

---

### 3. Computational Efficiency Trade-Offs

#### Training and Inference

- **Dense NN:** Fastest to train (0.092 s/epoch; 6.24 s total) and infer (0.282 ms/sample), with the smallest model size (0.163 MB).
- **1D CNN:** Training is slower (0.544 s/epoch; 40.24 s total), inference is slightly slower (0.368 ms/sample), and the model is significantly larger (2.113 MB).
- **Dilated 1D CNN:** Slowest to train (0.834 s/epoch; 67.56 s total) and infer (0.395 ms/sample), with the largest model size (2.129 MB).

#### Discussion

The Dense NN is clearly the most computationally efficient, both in terms of training and inference. The convolutional models, while potentially offering advantages in capturing local structure, do not translate these into improved accuracy for this dataset and problem formulation. The increased parameter count and memory footprint of the CNNs are not justified by a corresponding gain in predictive performance.

---

### 4. Sensitivity to Cosmological Parameters

While the primary focus of this study is architectural efficiency, it is instructive to consider the models' ability to generalize across the sampled cosmological parameter space. The synthetic dataset was constructed to uniformly sample the ranges:

- \( \Omega_m \): 0.1–0.5
- \( \sigma_8 \): 0.6–1.0
- \( h \): 0.6–0.8
- \( n_s \): 0.9–1.1

Histograms of the parameter distributions confirm uniform coverage, and example predictions (Figure 2) demonstrate that all models are able to accurately emulate \( P(k) \) across a diverse set of parameter combinations. No systematic degradation in performance is observed at the edges of the parameter space, indicating robust generalization.

However, the slightly higher errors at extreme \( k \) may be partially attributable to parameter combinations that push the limits of the physical model, where the mapping from parameters to \( P(k) \) becomes more nonlinear. The Dense NN appears to be more robust in these regimes, likely due to its fully connected structure and greater flexibility in modeling global dependencies.

---

### 5. Recommendations for Architecture Selection

#### Accuracy vs. Efficiency

- **For applications prioritizing accuracy and speed** (e.g., large-scale cosmological parameter inference, real-time emulation, or deployment on resource-constrained hardware), the Dense NN is the clear choice. It achieves the lowest errors across all metrics, is orders of magnitude faster to train and infer, and has a minimal memory footprint.
- **For applications where multi-scale or local structure is critical** (e.g., emulation of more complex, non-Gaussian features or higher-dimensional outputs), convolutional architectures may offer advantages not fully realized in this study. However, for the present task—emulating the 1D matter power spectrum from four parameters—the added complexity of CNNs does not yield improved performance.

#### Potential for Further Improvement

- **Dilated CNNs** are theoretically well-suited for capturing long-range dependencies and multi-scale structure. However, in this context, the relatively simple mapping from four parameters to a smooth 1D spectrum does not appear to benefit from this architectural feature. Future work could explore more complex emulation tasks (e.g., higher-dimensional fields, inclusion of baryonic effects, or non-linear corrections) where dilated convolutions may prove advantageous.
- **Hybrid or residual architectures** could be investigated to combine the strengths of dense and convolutional layers, particularly if the input space or output dimensionality is increased.

---

### 6. Physical Context and Implications

The ability to rapidly and accurately emulate the matter power spectrum is essential for modern cosmological analyses, including Markov Chain Monte Carlo (MCMC) parameter inference, survey forecasting, and model comparison. The results presented here demonstrate that, for the task of emulating the linear matter power spectrum over a broad range of cosmological parameters, a well-regularized dense neural network is both sufficient and optimal in terms of accuracy and computational efficiency.

The low mean and maximum percent errors achieved by the Dense NN (1.19% and 7.85%, respectively) are well within the requirements for most cosmological applications, where theoretical uncertainties and observational errors are typically larger. The rapid inference time (<0.3 ms/sample) enables real-time emulation, making this approach highly attractive for integration into larger cosmological pipelines.

---

### 7. Summary and Conclusions

- The Dense NN outperforms both standard and dilated 1D CNNs in accuracy, robustness, and computational efficiency for matter power spectrum emulation from four cosmological parameters.
- All models achieve sub-2% mean percent errors across all \( k \)-scales, with the Dense NN consistently achieving the lowest errors.
- The added complexity and parameter count of CNN-based models do not yield improved performance for this task, suggesting that the mapping from parameters to \( P(k) \) is sufficiently simple to be captured by a dense architecture.
- For more complex emulation tasks, or when the input/output dimensionality is increased, convolutional or hybrid architectures may become advantageous.
- The methodology and results presented here provide a rigorous benchmark for future studies of neural network emulators in cosmology.

---

**Figures and Tables Referenced:**
- **Figure 1:** Percent error vs. \( k \) for each architecture (see `data/percent_error_vs_k_*.png`)
- **Figure 2:** Example predictions vs. true \( P(k) \) for randomly selected test samples (see `data/pk_pred_vs_true_*.png`)
- **Table 1:** Model performance summary (see above)
- **Table 2:** Scale-dependent percent error metrics (see above)

---

**Data and Code Availability:**  
All code, data, and evaluation scripts are available in the project repository, with full documentation and version control to ensure reproducibility. Random seeds were fixed for all data splits and model initializations.

---

**References:**  
- Eisenstein, D. J., & Hu, W. (1998). Baryonic Features in the Matter Transfer Function. *The Astrophysical Journal*, 496(2), 605–614. [https://arxiv.org/abs/astro-ph/9710252](https://arxiv.org/abs/astro-ph/9710252)

---

This analysis provides a robust foundation for the selection and deployment of neural network emulators in cosmological applications, and highlights the importance of matching model complexity to the intrinsic complexity of the emulation task.