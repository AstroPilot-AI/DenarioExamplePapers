### Methodology for Mapping the Diversity of the Black Hole–Stellar Mass Relation

#### 1. Data Preparation and Integration

- **Data Loading:**  
  Load the full galaxy-level DataFrame (`galaxies_full_optimal.parquet`) and the catalog-level DataFrame (`catalog_params_optimal.parquet`) using efficient, chunked reading to minimize memory usage.
- **Merging Parameters:**  
  Ensure that each galaxy is associated with its parent catalog’s cosmological and feedback parameters. This is already present in the galaxy DataFrame, but cross-check for consistency.
- **Feature Selection:**  
  Retain only the necessary columns for analysis: `M_star`, `M_BH`, `SFR`, `catalog_number`, and the six cosmological/feedback parameters.

#### 2. Treatment of Galaxies with \( M_\mathrm{BH} = 0 \)

- **Rationale:**  
  A significant fraction of low-mass galaxies have \( M_\mathrm{BH} = 0 \), reflecting either physical absence or resolution limits. Including these as zeros in log-space regression would bias the results.
- **Approach:**  
  - **Primary Analysis:** Exclude galaxies with \( M_\mathrm{BH} = 0 \) from the regression analysis of the M_BH–M_star relation. This is justified by the focus on the scaling relation among galaxies with detected black holes, and by the large sample size in each mass bin.
  - **Secondary Analysis:** Quantify the black hole occupation fraction as a function of stellar mass and feedback parameters, to contextualize the main results and assess potential selection effects.
  - **Optional Robustness Check:** For completeness, perform censored regression (e.g., Tobit model) in a subset of catalogs to estimate the impact of non-detections, if computationally feasible.

#### 3. Stratification by Stellar Mass and SFR

- **Mass Binning:**  
  Divide galaxies into three stellar mass bins:  
  - Low: \( M_\mathrm{star} < 10^9\,M_\odot \)  
  - Intermediate: \( 10^9 \leq M_\mathrm{star} < 10^{10}\,M_\odot \)  
  - High: \( M_\mathrm{star} \geq 10^{10}\,M_\odot \)  
  This leverages the large, well-populated bins identified in the EDA and allows for regime-dependent analysis.
- **SFR Stratification:**  
  Optionally, further stratify by SFR bins (e.g., quiescent vs. star-forming) to test for secondary dependencies.

#### 4. Catalog-level Regression Analysis

- **Per-Catalog Fitting:**  
  For each catalog (simulation run), and within each stellar mass bin:
  - Select galaxies with \( M_\mathrm{BH} > 0 \).
  - Fit the relation \( \log_{10} M_\mathrm{BH} = \alpha + \beta \log_{10} M_\mathrm{star} \) using ordinary least squares (OLS) regression.
  - Record the best-fit slope (\( \beta \)), normalization (\( \alpha \)), and intrinsic scatter (standard deviation of residuals).
- **Scatter Estimation:**  
  Compute the standard deviation of the residuals in log-space as a measure of intrinsic scatter. If sample size in a bin is small (\(<20\)), flag the result as unreliable.

#### 5. Mapping Relation Parameters Across Parameter Space

- **Parameter Extraction:**  
  For each catalog, compile a table of:  
  - Catalog number  
  - Feedback and cosmological parameters  
  - Slope, normalization, and scatter of the M_BH–M_star relation in each mass bin
- **Multivariate Analysis:**  
  - Use multiple linear regression, random forest regression, or partial correlation analysis to quantify how the relation parameters (slope, normalization, scatter) depend on the feedback (A_SN1, A_SN2, A_AGN1, A_AGN2) and cosmological (Omega_m, sigma_8) parameters.
  - Assess the relative importance of each parameter using feature importance metrics or standardized regression coefficients.
  - Visualize trends using scatter plots, heatmaps, and partial dependence plots.

#### 6. Secondary Analyses

- **Black Hole Occupation Fraction:**  
  For each catalog and mass bin, compute the fraction of galaxies with \( M_\mathrm{BH} > 0 \) and analyze its dependence on feedback parameters.
- **SFR Dependence:**  
  Within each mass bin, test for residual trends in the M_BH–M_star relation parameters as a function of SFR, using partial correlation or regression with SFR as a covariate.

#### 7. Computational Optimization

- **Parallelization:**  
  - Distribute per-catalog regression analyses across 8 CPU cores using multiprocessing or joblib, processing batches of catalogs in parallel.
  - Each catalog is independent, enabling embarrassingly parallel computation.
- **Efficient Data Access:**  
  - Use chunked or on-demand loading of the galaxy DataFrame to avoid memory bottlenecks.
  - Precompute and cache intermediate results (e.g., galaxy selections per catalog and mass bin).
- **Batch Processing:**  
  - Process catalogs in batches (e.g., 100 at a time) to balance memory usage and CPU utilization.
- **Timing and Profiling:**  
  - Profile the runtime of a single catalog’s analysis to ensure that the full set can be completed within the 10-minute constraint. Adjust batch size or parallelization strategy as needed.

#### 8. Justification of Methodological Choices

- **Regression Approach:**  
  OLS regression in log-space is standard for scaling relations and justified by the strong correlation (\( r=0.86 \)) observed in the EDA.
- **Exclusion of \( M_\mathrm{BH} = 0 \):**  
  Exclusion avoids biasing the regression, given the physical and numerical ambiguity of zero-mass black holes. The large sample size in each bin ensures robust fits.
- **Mass Binning:**  
  The chosen bins reflect both physical regimes (SN- vs. AGN-dominated) and the population statistics from the EDA.
- **Multivariate Analysis:**  
  The use of regression and feature importance metrics enables quantitative assessment of the impact of each feedback and cosmological parameter.

#### 9. Summary Workflow

1. Load and merge data; select relevant features.
2. For each catalog and mass bin:
   - Exclude \( M_\mathrm{BH} = 0 \) galaxies.
   - Fit the M_BH–M_star relation; record slope, normalization, scatter.
   - Compute black hole occupation fraction.
3. Aggregate results across catalogs; merge with parameter table.
4. Analyze dependence of relation parameters on feedback and cosmological parameters using multivariate techniques.
5. Visualize and interpret trends, with attention to mass regime and SFR dependence.
6. Optimize computation via parallelization and batching to meet runtime constraints.

This methodology provides a robust, scalable framework for systematically mapping the diversity of the black hole–stellar mass relation across cosmological and feedback parameter space, leveraging the statistical power and parameter coverage of the dataset.