
### Methodology for Benchmarking 1D CNN and FCNN Architectures for Non-linear Matter Power Spectrum Emulation

#### 1. Data Generation

**a. Cosmological Parameter and Redshift Sampling**

- Employ Latin Hypercube Sampling (LHS) to generate a set of 500,000 unique samples in the 6-dimensional input space:  
  - Cosmological parameters:  
    - omega_b: [0.01933, 0.02533]  
    - omega_cdm: [0.08, 0.20]  
    - H0: [40, 100] (in km/s/Mpc)  
    - logA: [2.5, 3.5]  
    - n_s: [0.8, 1.2]  
  - Redshift: z ∈ [0, 1]  
- Each sample is a vector: (omega_b, omega_cdm, H0, logA, n_s, z).

**b. Power Spectrum Computation**

- For each sampled parameter set, use `classy_sz` to compute the non-linear matter power spectrum P(k, z) over the k-grid provided by `classy_sz`, covering k ∈ [10⁻⁴, 10] [1/Mpc].
- Store the k-grid for reference; ensure all spectra are evaluated on the same k-grid for consistency.


**c. Dataset Structuring and Splitting**

- Each data point consists of:
  - Input: 6-dimensional vector (cosmological parameters + redshift)
  - Output: 1D array of P(k, z) values (length = number of k-points from `classy_sz`)
- Randomly split the dataset into training (80%) and testing (20%) sets, ensuring stratification if possible to maintain coverage across the parameter space.

#### 2. Data Preprocessing

**a. Input Normalization**

- Apply min-max normalization to each input parameter, scaling to [0, 1] based on the sampled range.
- Store normalization parameters (min, max) for later use in inference.

**b. Output Transformation and Normalization**

- Transform the output spectra to log-log space:
  - Take the logarithm (base 10 or natural) of both k and P(k) values.
  - The neural networks will be trained to predict log₁₀ P(k) as a function of the input parameters.
- Optionally, standardize the output (subtract mean and divide by standard deviation for each k) based on the training set to improve training stability. Store these statistics for inverse transformation during evaluation.

**c. Data Storage**

- Store the preprocessed data in efficient binary formats (e.g., HDF5 or NumPy arrays) to facilitate fast loading during training.
- Maintain separate files for training and testing sets, including normalization metadata.

#### 3. Neural Network Training

**a. 1D CNN Architecture**

- Model architecture:
  - Input: 6-dimensional vector
  - Dense layer: output_dim = number of k-points, activation='relu'
  - Reshape: (output_dim, 1)
  - Conv1D: 64 filters, kernel size=3, activation='relu', padding='same'
  - Conv1D: 64 filters, kernel size=3, activation='relu', padding='same'
  - Flatten
  - Dense: 256 units, activation='relu'
  - Output: Dense layer, output_dim = number of k-points (no activation)
- Hyperparameters:
  - Batch size: 128 (tune as needed)
  - Optimizer: Adam
  - Learning rate: 1e-3 (tune as needed)
  - Loss function: Mean Squared Error (MSE)
  - Early stopping: monitor validation loss, patience = 20 epochs
  - Maximum epochs: 500
  - Optionally, use dropout (e.g., 0.1–0.2) after dense layers for regularization if overfitting is observed.

**b. FCNN Architecture**

- Model architecture:
  - Input: 6-dimensional vector
  - 4 hidden layers, each with 512 nodes, activation='swish'
  - Output: Dense layer, output_dim = number of k-points (no activation)
- Hyperparameters:
  - Same as for 1D CNN (batch size, optimizer, learning rate, loss, early stopping, max epochs)
  - Optionally, apply dropout (e.g., 0.1–0.2) after each hidden layer if overfitting is observed.

**c. Training Procedure**

- For both architectures:
  - Use the same training and validation splits for fair comparison.
  - Monitor training and validation loss; save the model with the lowest validation loss.
  - Log training curves for later analysis.

#### 4. Performance Evaluation

**a. Accuracy Metrics**

- Compute the following on the test set:
  - Mean Squared Error (MSE) between true and predicted log₁₀ P(k) across all k and all test samples.
  - Relative error: ΔP / P as a function of k and z, averaged over the test set.
- Stratify error analysis by k-range (e.g., linear regime, quasi-linear, non-linear) and by redshift bins to identify regime-specific performance.

**b. Inference Speed**

- Measure the wall-clock time required to predict the full P(k) array for a single input (cosmological parameters + z) for each architecture.
- Use a standardized computational environment (e.g., single CPU core or specified GPU).
- Report average and maximum inference time per spectrum over a batch of test samples.

**c. Diagnostic and Comparison Plots**

- Plot true vs. emulated P(k) (in log-log scale) for representative test cases at various redshifts and cosmologies.
- Plot error as a function of k and z for both architectures.
- Summarize results in tables and figures comparing accuracy and speed across regimes.

**d. Reporting**

- Present a comprehensive comparison of 1D CNN and FCNN architectures in terms of:
  - Overall and regime-specific accuracy
  - Inference speed
  - Diagnostic plots illustrating strengths and weaknesses in different physical regimes

---

This methodology ensures a rigorous, fair, and detailed comparison of 1D CNN and FCNN architectures for non-linear matter power spectrum emulation, providing actionable insights for cosmological emulator design and deployment.
