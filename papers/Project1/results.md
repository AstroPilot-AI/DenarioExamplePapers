<!-- filename: extensive_discussion_of_results.md -->
```markdown
<!-- filename: extensive_discussion_of_results.md -->
# Extensive Discussion of the Results

This document provides a comprehensive discussion and interpretation of the results obtained from the analysis pipeline for comparing the distributions of star formation and stellar assembly properties between Dataset A (fNL = 200) and Dataset B (fNL = –200). The discussion covers the statistical findings, their scientific implications, and potential caveats related to simulation limitations.

---

## 1. Summary of Statistical Findings

### 1.1. Descriptive Statistics and Distribution Comparisons

- **Star Formation Rates (GroupSFR and SubhaloSFR):**  
  Both GroupSFR and SubhaloSFR in the two datasets exhibit nearly identical means and standard deviations. The Kolmogorov-Smirnov (KS) tests return extremely low KS statistics with p-values near unity, indicating that the cumulative distributions of the star formation rates are statistically indistinguishable. Similarly, the Anderson-Darling (AD) tests yield AD statistics that are not significant (with p-values capped at 0.25%), suggesting no rejection of the null hypothesis that both samples are drawn from the same distribution.

- **Mass-Related Properties (SubhaloMass):**  
  The distributions of SubhaloMass also show close agreement between the two datasets. The small KS statistic and high p-value imply that the overall mass distributions are essentially the same, with only negligible differences in mean values and dispersion.

- **Photometric Properties (Various Stellar Photometric Bands):**  
  For the set of photometric features (U, B, V, K, g, r, i, z), the analysis revealed that their means and standard deviations were nearly identical between Dataset A and Dataset B. Both the KS and AD tests consistently indicated minimal differences, which is supported by the fact that the individual histograms and ratio plots are almost flat (ratio ≈ 1).

- **Derived SFR-to-Mass Ratio:**  
  When considering the ratio of SubhaloSFR to SubhaloMass, the statistical summaries again showed minimal differences between the two datasets. Both the KS and AD test results reaffirm that this derived metric is statistically equivalent across the two scenarios.

### 1.2. Visualization Insights

- **Histograms and Ratio Plots:**  
  The overlaid histograms for each property (plotted on logarithmic or linear scales depending on the variable’s dynamic range) visually corroborate the statistical tests: there are only slight or imperceptible differences between the distributions of the two datasets. The ratio plots, which display the point-by-point ratio of binned densities (fNL = 200 divided by fNL = –200), remain consistently near unity, further indicating that the relative differences are minimal.

---

## 2. Scientific Implications

### 2.1. Primordial Non-Gaussianity and Galaxy Formation

- **Subtle Effects on Bulk Properties:**  
  The negligible differences observed in star formation rates, mass distributions, and photometric properties imply that the bulk integrated properties of galaxies in these simulations are not dramatically sensitive to the level of primordial non-Gaussianity (between fNL = 200 and fNL = –200). This aligns with theoretical expectations that, while primordial non-Gaussianity can alter the initial conditions of structure formation, its imprint on averaged galaxy properties may be quite subtle, especially when other astrophysical processes (such as feedback and star formation regulation) dominate the later evolution.

- **Constraints on Cosmological Models:**  
  Since the measured differences are very minor, it may be challenging to use these particular observables as stringent constraints on fNL in future surveys. However, even subtle systematic trends if confirmed through higher resolution simulations or observational data could serve as a complementary probe to other methods, such as galaxy clustering and the cosmic microwave background (CMB) analyses.

### 2.2. Comparison With Theoretical Predictions

- **Consistency With Predictions:**  
  The results are in agreement with several theoretical models suggesting that while primordial non-Gaussianity can impact the halo bias and the formation of large-scale structures, its effect on individual galaxy properties (like SFR and luminosity) is less pronounced. Our analysis shows that the distributions of key observables remain nearly invariant under a moderate change in fNL.

- **Implications for Observational Strategies:**  
  These findings imply that future surveys that aim to constrain primordial non-Gaussianity might need to focus on higher-order statistics or on observables that are more directly sensitive to initial density fluctuations (such as clustering measures, weak lensing correlations, or rare-peak statistics) rather than solely relying on integrated properties like total star formation rates or average luminosities.

---

## 3. Consideration of Potential Systematic Effects

### 3.1. Numerical Artifacts and Simulation Limitations

- **Resolution and Sampling:**  
  One must consider that the lack of significant differences could partly be due to limitations in simulation resolution or in the sampling of the underlying galaxy population. If the simulations do not resolve key sub-grid physics or if the statistics are dominated by a large number of low-mass systems with modest star formation, subtle effects of fNL may be washed out.

- **Data Cleaning and Null Values:**  
  Our pipeline carefully cleaned the data by dropping NaN values on a per-feature basis. However, it is possible that such an approach might remove low signal-to-noise regions where small but systematic differences could reside. Future work might explore alternative imputation methods or adaptive binning techniques to ensure that genuine subtle differences are not inadvertently discarded.

- **Statistical Test Limitations:**  
  The p-values from the Anderson-Darling tests were capped (as the true p-values were larger than 0.25), indicating that the tests lack sensitivity in cases where the difference is extremely small. This limitation in statistical power should be kept in mind when interpreting the absence of significant differences.

### 3.2. Genuine Physical Effects vs. Artifacts

- **Real Physical Equivalence vs. Insensitivity:**  
  The near-equality of the distributions across the two different fNL scenarios may indicate a genuine physical insensitivity of these integrated observables to moderate changes in the primordial non-Gaussianity parameter. Alternatively, it could also be an artifact of the particular simulation setup, including the implemented feedback models and the initial conditions.

- **Need for Complementary Probes:**  
  To robustly disentangle physical effects from numerical artifacts, it is advisable to compare these findings with complementary observational probes and to conduct similar analyses on different simulation suites or on simulations with a wider range of fNL values.

---

## 4. Conclusion and Future Directions

The extensive analysis performed through our pipeline shows that key galaxy properties—such as star formation rates, mass indicators, and photometric magnitudes—do not exhibit significant differences between the two simulated non-Gaussianity scenarios (fNL = 200 vs. fNL = –200). This suggests that, for the scales and timeframes probed by these simulations, the observable impact of primordial non-Gaussianity on these properties is either very subtle or masked by other dominating astrophysical processes.

**Future Work:**

- **Enhanced Simulation Studies:**  
  Running simulations with higher resolution or with a broader range of fNL values might help in detecting and quantifying any subtle effects more clearly.

- **Alternative Observables:**  
  Investigating higher-order statistics (e.g., skewness, kurtosis) or spatial correlation functions may provide additional insights into how primordial non-Gaussianity affects galaxy clustering and bias.

- **Multi-Probe Analysis:**  
  Combining these bulk properties with other observational probes (such as lensing or redshift-space distortions) could help break degeneracies and improve constraints on primordial non-Gaussianity.

- **Systematic Error Mitigation:**  
  Further work is needed to refine data cleaning and statistical testing methodologies in order to improve sensitivity to small differences while mitigating potential artifacts.

In summary, while our current analysis finds that the integrated properties of galaxies simulated under two different fNL scenarios are remarkably similar, this outcome provides a valuable benchmark. It suggests that more sensitive or complementary observational methods may be required to detect the subtle imprints of primordial non-Gaussianity on galaxy formation and evolution.

```