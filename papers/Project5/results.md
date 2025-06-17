## Results and Interpretation: Mapping the Diversity of the Black Hole–Stellar Mass Relation Across Cosmological and Feedback Parameter Space

### 1. Overview and Methodological Recap

This section presents a comprehensive analysis of the diversity in the black hole–stellar mass ($M_\mathrm{BH}$–$M_\mathrm{star}$) relation across a wide range of cosmological and feedback parameter space, leveraging a suite of 1,000 simulated galaxy catalogs. Each catalog samples a unique combination of cosmological parameters ($\Omega_m$, $\sigma_8$) and feedback parameters ($A_\mathrm{SN1}$, $A_\mathrm{SN2}$, $A_\mathrm{AGN1}$, $A_\mathrm{AGN2}$), enabling a systematic exploration of how the scaling relation’s slope, normalization, and scatter respond to changes in the underlying physics.

The analysis proceeds by fitting the relation $\log_{10} M_\mathrm{BH} = \alpha + \beta \log_{10} M_\mathrm{star}$ within three stellar mass bins (low: $<10^9\,M_\odot$, intermediate: $10^9$–$10^{10}\,M_\odot$, high: $>10^{10}\,M_\odot$) for each catalog, excluding galaxies with $M_\mathrm{BH}=0$ to focus on the scaling among black hole hosts. The resulting best-fit parameters (slope $\beta$, normalization $\alpha$, and intrinsic scatter) are then mapped as functions of the six catalog parameters using both multivariate linear regression and random forest analysis, with uncertainties estimated via bootstrapping. The black hole occupation fraction is also quantified as a function of mass and feedback.

### 2. Diversity of the $M_\mathrm{BH}$–$M_\mathrm{star}$ Relation Across Mass Bins

#### 2.1. Distributions of Slope, Normalization, and Scatter

**Low Mass Bin ($M_\mathrm{star}<10^9\,M_\odot$):**
- **Slope ($\beta$):** Mean $0.20$, std $0.09$, range $[-0.07, 0.58]$.
- **Normalization ($\alpha$):** Mean $4.26$, std $0.72$, range $[1.24, 6.54]$.
- **Scatter:** Mean $0.11$ dex, std $0.05$ dex.
- **Occupation Fraction:** Mean $0.72$, std $0.13$.

**Intermediate Mass Bin ($10^9$–$10^{10}\,M_\odot$):**
- **Slope ($\beta$):** Mean $1.06$, std $0.42$, range $[0.02, 1.84]$.
- **Normalization ($\alpha$):** Mean $-3.53$, std $3.80$, range $[-10.6, 6.05]$.
- **Scatter:** Mean $0.30$ dex, std $0.06$ dex.
- **Occupation Fraction:** Mean $0.85$, std $0.08$.

**High Mass Bin ($>10^{10}\,M_\odot$):**
- **Slope ($\beta$):** Mean $1.22$, std $0.30$, range $[0.03, 2.07]$.
- **Normalization ($\alpha$):** Mean $-4.85$, std $3.39$, range $[-14.2, 7.64]$.
- **Scatter:** Mean $0.40$ dex, std $0.14$ dex.
- **Occupation Fraction:** Mean $0.96$, std $0.03$.

**Interpretation:**  
The $M_\mathrm{BH}$–$M_\mathrm{star}$ relation exhibits substantial diversity across catalogs, with the slope and normalization varying by factors of several, especially in the intermediate and high mass bins. The scatter increases with stellar mass, and the occupation fraction rises from $\sim$70% at low mass to nearly unity at high mass. These trends are visualized in the histograms and scatter plots (see, e.g., `dist_beta_high_84_...png`, `dist_alpha_intermediate_43_...png`).

#### 2.2. Physical Regimes

- **Low Mass Regime:** The shallow mean slope ($\sim$0.2) and high normalization reflect a regime where black hole growth is inefficient or stochastic, and feedback processes (especially supernova-driven) dominate the baryon cycle.
- **Intermediate/High Mass Regimes:** The slope approaches or exceeds unity, consistent with the canonical $M_\mathrm{BH}$–$M_\mathrm{star}$ relation observed in massive galaxies. The increased scatter and normalization diversity suggest a strong sensitivity to feedback and cosmological parameters.

### 3. Dependence on Feedback and Cosmological Parameters

#### 3.1. Multivariate Regression and Feature Importance

##### 3.1.1. Slope ($\beta$)

- **Low Mass Bin:** The most influential parameters are $A_\mathrm{SN1}$ (SN wind energy per SFR; standardized coefficient $+0.05$, RF importance $0.41$) and $A_\mathrm{AGN1}$ (AGN feedback energy per accretion; $+0.04$, RF $0.29$). Both positively correlate with slope, indicating that stronger feedback steepens the relation at low mass. Cosmological parameters have weaker effects.
- **Intermediate/High Mass Bins:** $A_\mathrm{AGN1}$ becomes dominant (intermediate: $+0.37$, RF $0.88$; high: $-0.22$, RF $0.69$), with a positive effect in the intermediate bin and a negative effect in the high bin. This suggests a transition from AGN-driven black hole growth at intermediate mass to AGN-driven suppression at high mass. $A_\mathrm{SN1}$ and cosmological parameters have smaller or inconsistent effects.

##### 3.1.2. Normalization ($\alpha$)

- **Low Mass Bin:** $A_\mathrm{SN1}$ and $A_\mathrm{AGN1}$ have strong negative coefficients ($-0.38$ and $-0.32$), indicating that stronger feedback lowers the normalization, i.e., black holes are less massive at fixed stellar mass. $\Omega_m$ also has a negative effect.
- **Intermediate/High Mass Bins:** $A_\mathrm{AGN1}$ is the dominant driver (intermediate: $-3.32$, high: $+2.52$), with the sign flip again reflecting a regime change. $\sigma_8$ and $\Omega_m$ also contribute, with higher matter density and power spectrum normalization generally lowering $\alpha$.

##### 3.1.3. Scatter

- **Low Mass Bin:** $A_\mathrm{SN1}$ is the primary driver of increased scatter (coefficient $+0.04$, RF $0.73$), consistent with the expectation that stochastic SN feedback introduces diversity in black hole growth histories.
- **Intermediate/High Mass Bins:** $A_\mathrm{AGN1}$ and $A_\mathrm{SN1}$ both contribute, with $A_\mathrm{AGN1}$ increasing scatter at intermediate mass (RF $0.44$) and $A_\mathrm{SN1}$ at high mass (RF $0.31$).

##### 3.1.4. Partial Dependence and 2D Interactions

- **Partial dependence plots** (e.g., `pdp_Slope($\beta$)_high_113_...png`) show that increasing $A_\mathrm{AGN1}$ or $A_\mathrm{SN1}$ can both steepen the slope and increase the scatter, but the effect is nonlinear and mass-dependent.
- **2D heatmaps** (e.g., `heatmap_beta_A_AGN1_A_AGN2_high_105_...png`) reveal that the joint influence of AGN1 and AGN2 is especially pronounced at high mass, with the steepest slopes and highest normalizations occurring for high AGN1 and low AGN2.

#### 3.2. Cosmological Parameters

- **$\Omega_m$ and $\sigma_8$** have secondary but non-negligible effects, especially on normalization and scatter. Higher $\Omega_m$ generally lowers normalization and increases scatter, while higher $\sigma_8$ can either increase or decrease normalization depending on mass bin.

#### 3.3. Summary of Feature Importance

- **Low Mass:** SN feedback dominates.
- **Intermediate Mass:** AGN feedback (especially AGN1) dominates, with some SN contribution.
- **High Mass:** AGN feedback remains dominant, but the sign of its effect on slope and normalization reverses, indicating a shift from black hole fueling to quenching.

### 4. Black Hole Occupation Fraction

#### 4.1. Trends with Mass and Feedback

- **Occupation fraction** rises from $\sim$0.72 at low mass to $\sim$0.96 at high mass, with significant catalog-to-catalog scatter (see occupation fraction plots, e.g., `occfrac_A_SN1_high_122_...png`).
- **Feedback dependence:** Stronger SN feedback ($A_\mathrm{SN1}$) reduces the occupation fraction at low and intermediate mass, consistent with the suppression of black hole formation or growth in shallow potential wells. AGN feedback has a weaker effect on occupation at low mass but becomes more important at high mass.

#### 4.2. Implications for Black Hole Seeding

- The incomplete occupation at low mass, and its sensitivity to feedback, supports models in which black hole seeding is inefficient or stochastic in low-mass halos, and can be suppressed by energetic feedback. This is consistent with recent observational inferences of low occupation fractions in dwarf galaxies (e.g., Greene et al. 2020, Nature Astronomy, 4, 154).

### 5. Secondary Dependencies and SFR

- **SFR dependence:** While the primary analysis did not stratify by SFR, the weak correlation between SFR and $M_\mathrm{star}$ ($r=0.10$) and the moderate effect of feedback parameters on both SFR and $M_\mathrm{BH}$ suggest that residual trends with SFR are likely subdominant compared to the direct effects of feedback and mass. Future work could explicitly include SFR as a covariate.

### 6. Comparison with Observational Constraints

- **Observed $M_\mathrm{BH}$–$M_\mathrm{star}$ relation:** The mean slopes ($\beta\sim1.1$–$1.2$ at intermediate/high mass) and scatters ($\sim$0.3–0.4 dex) are broadly consistent with local observations (e.g., Kormendy & Ho 2013, ARA&A, 51, 511; McConnell & Ma 2013, ApJ, 764, 184), though the normalization and scatter exhibit greater diversity in the simulations, especially at low mass.
- **Occupation fraction:** The simulated occupation fractions at high mass are consistent with the near-unity values inferred for massive galaxies, while the lower and more variable occupation at low mass matches the emerging observational picture for dwarfs and low-mass systems.

### 7. Synthesis and Physical Interpretation

#### 7.1. Feedback as the Primary Driver of Diversity

The results demonstrate that the diversity in the $M_\mathrm{BH}$–$M_\mathrm{star}$ relation is primarily driven by the strength and mode of feedback, with SN feedback dominating at low mass and AGN feedback at high mass. The transition between these regimes is marked by a shift in the sign and magnitude of the feedback coefficients, and is accompanied by changes in the occupation fraction and scatter.

#### 7.2. Cosmological Context

While feedback is the dominant driver, cosmological parameters modulate the normalization and scatter, particularly in the intermediate and high mass bins. This highlights the importance of considering both baryonic and cosmological physics in modeling black hole–galaxy coevolution.

#### 7.3. Implications for Galaxy Evolution

The broad range of slopes, normalizations, and scatters observed across the simulated catalogs underscores the potential for significant diversity in the $M_\mathrm{BH}$–$M_\mathrm{star}$ relation in the real universe, especially in regimes where feedback is strong or stochastic. The results suggest that observed outliers and the intrinsic scatter in the scaling relation may be natural consequences of variations in feedback efficiency and cosmological context.

#### 7.4. Black Hole Seeding and Occupation

The sensitivity of the occupation fraction to feedback at low mass provides a direct link between black hole seeding models and the observable demographics of low-mass galaxies. The results support scenarios in which energetic feedback can suppress black hole formation or growth in shallow potential wells, leading to incomplete occupation and a wide diversity of black hole masses at fixed stellar mass.

### 8. Conclusions

This analysis provides the first systematic mapping of the diversity in the $M_\mathrm{BH}$–$M_\mathrm{star}$ relation across a well-sampled cosmological and feedback parameter space. The key findings are:

- The slope, normalization, and scatter of the $M_\mathrm{BH}$–$M_\mathrm{star}$ relation vary substantially across catalogs, with feedback parameters (especially $A_\mathrm{SN1}$ and $A_\mathrm{AGN1}$) as the primary drivers.
- The black hole occupation fraction increases with stellar mass and is strongly suppressed by SN feedback at low mass.
- Cosmological parameters modulate the normalization and scatter, but play a secondary role compared to feedback.
- The simulated relations are broadly consistent with observational constraints at high mass, but predict greater diversity and lower occupation at low mass, in line with recent observations.
- The results highlight the importance of feedback physics in shaping the coevolution of black holes and galaxies, and provide a framework for interpreting the observed diversity in black hole scaling relations.

All key statistics, regression coefficients, feature importances, and visualizations are available in the supplementary figures and tables (see data directory for filenames). This work lays the foundation for future studies incorporating additional observables (e.g., SFR, environment) and for direct comparison with upcoming large-scale surveys of black holes in low-mass galaxies.