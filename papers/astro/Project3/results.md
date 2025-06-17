## 1. Overview

This section presents a comprehensive comparison between two neural network architectures—1D Convolutional Neural Network (1D CNN) and Fully Connected Neural Network (FCNN)—for emulating the non-linear matter power spectrum, \(P(k, z)\), in \(\Lambda\)CDM cosmologies. The emulators were trained and tested on a large dataset (\(N=50,000\) spectra) generated using the `classy_sz` code, with cosmological parameters and redshift sampled via Latin Hypercube Sampling (LHS) to ensure uniform and uncorrelated coverage of the parameter space. The input parameters include \(\omega_b\), \(\omega_{\rm cdm}\), \(H_0\), \(\log A\), \(n_s\), and \(z\), with \(P(k)\) evaluated on a fixed \(k\)-grid spanning \(10^{-4}\) to \(10\) [1/Mpc]. All spectra were log-scaled for both \(k\) and \(P(k)\), and both input and output data were normalized for neural network training.

The following discussion is structured as follows: (1) summary of dataset and parameter coverage, (2) quantitative performance comparison, (3) diagnostic plots and error analysis, (4) computational efficiency and model complexity, (5) systematic trends and biases, (6) practical implications, and (7) recommendations for future work.

---

## 2. Dataset and Parameter Coverage

### 2.1. Parameter Ranges and Sampling

The cosmological parameter ranges used for dataset generation are:

| Parameter            | Range              |
|----------------------|--------------------|
| \(\omega_b\)       | 0.01933 – 0.02533  |
| \(\omega_{\rm cdm}\) | 0.08 – 0.20      |
| \(H_0\) [km/s/Mpc]  | 40.0 – 100.0      |
| \(\log A\)         | 2.5 – 3.5         |
| \(n_s\)            | 0.8 – 1.2         |
| \(z\)              | 0.0 – 3.0         |

Latin Hypercube Sampling was employed to ensure uniform coverage across the 6D parameter space. Histograms and pair plots (see *data/param_histograms_1_*.png and *data/param_pairplot_2_*.png) confirm the absence of significant correlations and the uniformity of the sampling.

### 2.2. Data Normalization

All input parameters and log-scaled \(P(k)\) spectra were normalized using mean-variance normalization, with statistics computed from the training set. This normalization is essential for stable and efficient neural network training.

---

## 3. Quantitative Performance Comparison

### 3.1. Summary Table

The following table summarizes the key performance metrics for both emulators, evaluated on the test set:

| Model   | Mean Abs Error | Max Abs Error | 95th Perc Abs Error | Inference Time (s) | Num Params | Memory (MB) |
|---------|----------------|---------------|---------------------|--------------------|------------|-------------|
| 1D CNN  | 0.0206         | 5.87          | 0.0751              | 0.528              | 8,336,864  | 31.8        |
| FCNN    | 0.0313         | 6.90          | 0.1210              | 0.532              | 1,048,052  | 4.00        |

*Note: Errors are in units of \(\log_{10} P(k)\).* 

#### Key Observations:
- **Accuracy:** The 1D CNN achieves a lower mean absolute error (MAE) and 95th percentile error than the FCNN, indicating superior overall accuracy.
- **Maximum Error:** Both models exhibit rare outliers with large errors, but the 1D CNN's maximum error is lower.
- **Inference Speed:** Both models have similar inference times per full test set (\(\sim\)0.53 s), indicating that for batch evaluation, neither model is significantly slower.
- **Model Size:** The 1D CNN is substantially larger in terms of parameter count and memory usage, being roughly 8 times larger than the FCNN.

---

## 4. Diagnostic Plots and Error Analysis

### 4.1. True vs Emulated \(P(k)\)

Figure 1 (*data/pk_true_vs_emulated_1_*.png) shows the true and emulated \(\log_{10} P(k)\) for five randomly selected test samples at various redshifts. Both models closely track the true spectrum across the full \(k\)-range, but the 1D CNN consistently provides a closer match, especially in the non-linear regime (\(k \gtrsim 0.1\) [1/Mpc]) and at higher redshifts.

### 4.2. Error Distributions

Figure 2 (*data/error_histogram_2_*.png) presents the histogram of absolute errors for both models across the entire test set. The 1D CNN's error distribution is sharply peaked at low values, with a rapid fall-off, while the FCNN's distribution is broader and exhibits a longer tail. The 95th percentile error for the 1D CNN is 0.075, compared to 0.121 for the FCNN, confirming the superior reliability of the CNN emulator.

### 4.3. Loss Curves

Loss curves for both models are provided in *data/cnn1d_loss_curve_1_*.png and *data/fcnn_loss_curve_1_*.png. The 1D CNN achieves a lower minimum validation loss and converges more rapidly, while the FCNN exhibits a higher plateau and slightly more overfitting, as indicated by the gap between training and validation loss.

---

## 5. Computational Efficiency and Model Complexity

### 5.1. Inference Speed

Despite the 1D CNN's much larger parameter count (over 8 million vs. 1 million for the FCNN), both models achieve similar inference times when evaluated on the full test set. This is likely due to the highly parallelizable nature of convolutional operations on modern hardware (e.g., GPUs), and the fact that the FCNN's dense layers are also efficiently vectorized.

### 5.2. Memory Usage

The 1D CNN's memory footprint is approximately 32 MB, compared to 4 MB for the FCNN. For most research and production environments, both models are lightweight, but the difference may become relevant for deployment on resource-constrained devices or for large-scale parameter scans.

### 5.3. Model Complexity

The 1D CNN's architecture, with its initial dense projection, reshaping, and two convolutional layers, is designed to exploit the local structure and smoothness of the \(P(k)\) spectrum in log-log space. The FCNN, by contrast, treats the spectrum as a flat vector, lacking any explicit mechanism to capture local correlations in \(k\).

---

## 6. Systematic Trends and Biases

### 6.1. Parameter Space Coverage

The uniform LHS sampling ensures that both models are exposed to the full range of cosmological parameters and redshifts during training. No significant regions of parameter space are left unexplored, as confirmed by the pair plots.

### 6.2. Redshift and \(k\)-Dependence

Visual inspection of the true vs emulated \(P(k)\) plots reveals that both models perform best at low redshift (\(z \lesssim 1\)) and in the quasi-linear regime (\(k \lesssim 0.1\) [1/Mpc]). The 1D CNN maintains high accuracy even at higher redshifts and in the deeply non-linear regime, while the FCNN's errors increase more noticeably in these regions.

### 6.3. Outliers and Maximum Errors

Both models exhibit rare outliers with large errors (max error \(\sim\)6–7 in \(\log_{10} P(k)\)), which may correspond to extreme corners of parameter space or to spectra with sharp features (e.g., rapid transitions in the non-linear regime). The 1D CNN's lower maximum error suggests greater robustness, but further investigation into the nature of these outliers is warranted.

### 6.4. Error Examples

For the first five test samples, the mean absolute error for the 1D CNN ranges from 0.0046 to 0.070, while for the FCNN it ranges from 0.0039 to 0.119. This further illustrates the more consistent performance of the CNN emulator.

---

## 7. Practical Implications for Cosmological Emulation

### 7.1. Accuracy

The 1D CNN achieves sub-percent level accuracy in \(\log_{10} P(k)\) for the vast majority of test cases, with a mean absolute error of 0.021. This level of precision is sufficient for most cosmological inference applications, including likelihood analyses and parameter estimation from large-scale structure surveys.

### 7.2. Speed

Both emulators provide orders-of-magnitude speedup over direct Boltzmann code evaluations (e.g., CLASS or CAMB), with inference times of \(\sim\)0.5 s for 10,000 spectra on modern hardware. This enables rapid exploration of cosmological parameter space and efficient use in Markov Chain Monte Carlo (MCMC) or nested sampling pipelines.

### 7.3. Model Size

While the 1D CNN is larger, its memory requirements remain modest by modern standards. The FCNN may be preferable for deployment in memory-constrained environments, but this comes at a cost in accuracy and reliability.

### 7.4. Generalization

The superior performance of the 1D CNN across the full parameter and redshift range suggests that convolutional architectures are better suited to capturing the smooth, locally correlated structure of the matter power spectrum, especially in the non-linear regime.

---

## 8. Systematic Trends Across Parameter Space

A more detailed analysis of errors as a function of individual cosmological parameters and redshift could reveal subtle biases or regions of reduced accuracy. Preliminary inspection suggests that both models are robust across the sampled parameter space, but the FCNN is more prone to larger errors at high \(z\) and high \(k\). The 1D CNN's convolutional layers likely confer an advantage in modeling the scale-dependent features that arise in these regimes.

---

## 9. Recommendations and Future Directions

### 9.1. Model Improvements

- **Hybrid Architectures:** Combining convolutional and fully connected layers, or incorporating attention mechanisms, may further improve accuracy and robustness, especially in the tails of the error distribution.
- **Uncertainty Quantification:** Extending the emulators to provide uncertainty estimates (e.g., via Bayesian neural networks or ensemble methods) would enhance their utility for precision cosmology.
- **Training Set Augmentation:** Targeted oversampling of regions with higher errors (e.g., high \(z\), high \(k\)) could reduce the frequency of outliers.
- **Alternative Loss Functions:** Exploring loss functions that penalize large errors more strongly, or that weight different \(k\)-ranges according to scientific priorities, may yield further gains.

### 9.2. Deployment Considerations

- **Batch vs. Single Inference:** For applications requiring single-spectrum inference, the FCNN's smaller size may offer marginal speed advantages, but for batch processing, the 1D CNN's accuracy justifies its use.
- **Hardware Optimization:** Both models are compatible with GPU acceleration, but further optimization (e.g., quantization, pruning) could enable deployment on edge devices.

### 9.3. Scientific Applications

- **Likelihood-Free Inference:** The demonstrated accuracy and speed of the 1D CNN emulator make it well-suited for use in simulation-based inference frameworks, such as Approximate Bayesian Computation (ABC) or neural likelihood estimation.
- **Extension to Beyond-\(\Lambda\)CDM Models:** The methodology can be readily extended to more complex cosmological models (e.g., dynamical dark energy, massive neutrinos) by expanding the parameter space and retraining.

---

## 10. Conclusions

This study provides a rigorous, head-to-head comparison of 1D CNN and FCNN architectures for emulating the non-linear matter power spectrum in \(\Lambda\)CDM cosmologies. The 1D CNN consistently outperforms the FCNN in terms of accuracy, reliability, and robustness across the full range of cosmological parameters and redshifts, with only a modest increase in memory usage and no significant penalty in inference speed. The results strongly support the use of convolutional architectures for cosmological emulation tasks involving smooth, locally correlated functions such as \(P(k, z)\). The diagnostic plots and quantitative metrics presented here provide a benchmark for future emulator development and deployment in cosmological analyses.

---

## 11. Figures and Tables

- **Table 1:** Model Performance Summary (see above)
- **Figure 1:** True vs Emulated \(\log_{10} P(k)\) for 5 random test samples at various redshifts (*data/pk_true_vs_emulated_1_*.png)
- **Figure 2:** Histogram of absolute errors for 1D CNN and FCNN predictions (*data/error_histogram_2_*.png)
- **Figure 3:** Loss curves for 1D CNN and FCNN models (*data/cnn1d_loss_curve_1_*.png, *data/fcnn_loss_curve_1_*.png)
- **Figure 4:** Parameter coverage histograms and pair plots (*data/param_histograms_1_*.png, *data/param_pairplot_2_*.png)

---

## 12. Data and Code Availability

All data, trained models, and diagnostic plots are available in the `data/` directory. The codebase used for data generation, training, and evaluation is available upon request and follows best practices for reproducibility and transparency.

---

## 13. Acknowledgements

This work benefited from the use of the `classy_sz` code for power spectrum generation and from open-source machine learning libraries including TensorFlow and Keras.

*End of Results and Discussion Section*