<!-- filename: methods_for_selected_idea.md -->
```markdown
<!-- filename: methods_for_selected_idea.md -->
# Methodology for "Effects on Star Formation and Stellar Assembly in Different fNL Scenarios"

This document outlines a detailed methodology for implementing the selected research project idea, which examines how differing primordial non-Gaussianity values (fNL = 200 vs. fNL = –200) impact star formation and stellar assembly processes. The following sections describe the data preprocessing steps, statistical analysis framework, and visualization techniques that will be used, as well as identify key features and the physical rationale behind their expected differences.

---

## 1. Data Preprocessing

### 1.1. Data Import and Verification
- **Data Sources:**  
  - Dataset A (fNL = 200)  
  - Dataset B (fNL = –200)
- **Actions:**
  - Load both datasets from their respective pickle files.
  - Verify the integrity and structure of each dataset (using `.info()` and `.describe()`) to confirm feature presence and appropriate data types.
  - Compare non-null counts between datasets for features of interest to understand potential sample size limitations.

### 1.2. Cleaning and Filtering
- **Handling Missing Values:**
  - For key features such as GroupSFR, SubhaloSFR, and photometric properties (SubhaloStellarPhotometrics_U, B, V, K, g, r, i, z), remove or filter out rows that contain NaN values.
  - Optionally, log the fraction of missing data to assess if any additional imputation is necessary (though caution is advised since imputation might mask distribution differences).
- **Normalizing Units and Scales:**
  - Ensure that units and scales are consistently applied across both datasets.
  - Consider transforming highly skewed variables (e.g., SFR, SubhaloMass) with a logarithmic scaling to compress wide dynamic ranges.
- **Data Subsetting:**
  - Focus primarily on the following features:
    - **Star Formation and Mass Indicators:**  
      - *GroupSFR* (aggregate SFR at the group level)
      - *SubhaloSFR* (star formation rate at the subhalo level)
      - *SubhaloMass* (correlates with galaxy potential and star formation efficiency)
    - **Photometric Properties:**  
      - *SubhaloStellarPhotometrics_U, B, V, K, g, r, i, z*  
        (these will be used to derive luminosity, color indices, and assess stellar assembly history)
  - Optionally, create derived quantities such as SFR-to-mass ratios that might expose differences in star formation efficiency.

---

## 2. Statistical Analysis

### 2.1. Distribution Comparison
- **Descriptive Statistics:**
  - Compute the mean, median, standard deviation, and percentiles for SFR and photometric features in both datasets.
  - Compare central tendencies and spread for each key feature.
- **Statistical Tests for Distribution Differences:**
  - **Kolmogorov-Smirnov (KS) Test:**
    - Apply the KS test to compare the cumulative distributions of selected features (SFRs and photometric magnitudes) between datasets A and B.
  - **Anderson-Darling Test:**
    - Utilize the Anderson-Darling test to assess if the samples are drawn from the same distribution, offering enhanced sensitivity to differences in the tails.
  - **Additional Tests:**
    - Consider other non-parametric tests if necessary (e.g., Mann-Whitney U) to further validate findings.

### 2.2. Metric and Ratio Analysis
- **Ratios and Scaling:**
  - Compute ratios such as *SubhaloSFR/SubhaloMass* for individual subhalos, then compare their distributions.
  - Create bin-by-bin ratio plots that capture relative differences between the two datasets.
- **Color Indices:**
  - Derive color indices (e.g., g - r, U - B) from the photometric features.
  - Analyze any systematic shifts in these indices, as differences in colors can be linked to variations in stellar populations and star formation histories.

### 2.3. Physical Mechanisms
- **Physical Interpretation:**
  - The primordial non-Gaussianity parameter (fNL) influences the initial density fluctuation spectrum:
    - With a positive fNL (200), one might expect more pronounced over-densities in the early universe, potentially leading to earlier or more intense star formation episodes.
    - With a negative fNL (–200), the initial density field may be less extreme, possibly resulting in a more delayed or lower star formation rate.
  - These early conditions could imprint differences in the mass assembly and the stellar populations, reflected by variations in SFR, luminosities, and color distributions.
- **Target Features:**
  - Emphasize investigations on:
    - **SubhaloSFR and GroupSFR:** Expected to show potential systematic shifts due to varying star formation intensities.
    - **Photometric Magnitudes:** Differences in brightness and color can trace variations in stellar ages and metallicity distributions, which are indirectly influenced by early star formation history affected by fNL.

---

## 3. Visualization Techniques

### 3.1. Histograms and Density Plots
- **Histograms:**
  - Plot histograms for both datasets for key features.
  - Use logarithmic scales for SFR and mass-related features to capture their wide dynamic ranges; use linear scales for photometric magnitudes to preserve color information.
  - Ensure dynamic binning strategies to capture both central tendencies and tail behaviors.
- **Density Plots:**
  - Generate kernel density estimates (KDE) for smoother visualizations of the distribution differences.

### 3.2. Scatter and Ratio Plots
- **Scatter Plots:**
  - Create scatter plots correlating SFR and mass, as well as plotting color indices against SFR to identify trends.
  - Include separate markers or colors to distinguish between the positive and negative fNL datasets.
- **Ratio Plots:**
  - Construct ratio plots by taking the ratio of binned counts (or densities) in datasets A and B. This will highlight regions where the differences are most pronounced.
  - Use log-log scales where applicable to visualize variations spanning multiple orders of magnitude.

### 3.3. Statistical Summary Visuals
- **Box Plots/Violin Plots:**
  - Display box or violin plots to visually compare the spread and central tendency of each feature between the two simulations.
  - These plots help in visualizing outliers and the overall distribution shape.

---

## 4. Implementation Workflow

### Workflow Overview:
1. **Data Loading and Cleaning:** Load both datasets, filter and clean the data for target features.
2. **Feature Engineering:** Create derived quantities such as SFR-to-mass ratios and color indices.
3. **Exploratory Data Analysis (EDA):** Compute descriptive statistics and visualize distributions via histograms, density plots, and scatter plots.
4. **Statistical Testing:** Perform KS, Anderson-Darling, and possibly Mann-Whitney U tests to rigorously compare the distributions.
5. **Visualization and Ratio Analysis:** Develop ratio plots and comparative visuals (box plots, violin plots) to identify and highlight differences.
6. **Interpretation:** Map the statistical and visual findings to physical mechanisms influenced by primordial non-Gaussianity.

### Tools and Technologies:
- **Programming Language:** Python
- **Libraries:**  
  - Pandas for data manipulation  
  - NumPy for numerical operations and vectorized computations  
  - SciPy for statistical tests  
  - Matplotlib/Seaborn for plotting  
  - Optionally, Dask for performance optimization with large arrays

---

## Summary

In summary, the methods for this project involve:
- **Data Preprocessing:** Clean the datasets and focus on star formation and photometric features.
- **Statistical Analysis:** Use descriptive statistics and non-parametric tests (KS, Anderson-Darling) to compare the distributions of SFR and stellar magnitudes.
- **Visualization Techniques:** Employ histograms, ratio plots, scatter plots, and box/violin plots to capture and compare the dynamic ranges and subtle shifts in the data.
- **Physical Interpretation:** Connect observed differences to the impact of primordial non-Gaussianity on early universe conditions, influencing star formation efficiency and stellar assembly.

This integrated methodology should provide a robust framework for investigating how different primordial non-Gaussianity scenarios imprint themselves on the observable properties of galaxies.
```