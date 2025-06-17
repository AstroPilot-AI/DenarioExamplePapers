
The data is located under:
root=/Users/boris/CMBAgents/AstroPilot/project_data/data/

Galaxy and Catalog DataFrames: Structure and Usage Guide

This data is from CAMELS (Cosmology and Astrophysics with MachinE 
Learning Simulations). CAMELS have been designed to achieve these goals by making using of machine learning techniques. With more than 4,000 numerical simulations, both N-body and (magneto-)hydrodynamic, CAMELS follow the evolution of more than 100 billion dark matter particles and fluid elements in a combined volume of ~ (400 Mpc/h). CAMELS span thousands of different cosmological and astrophysical models. 

1. Full Galaxy DataFrame ('galaxies_full_optimal.parquet')

   - Each row: one galaxy at z=0 from one of 1000 simulated catalogs.

   - Columns: 17 galaxy features, catalog_number, and 6 cosmological/astrophysical parameters.

   - Shape: (720548, 24)

   - File format: Parquet (gzip compressed)


2. Catalog-level DataFrame ('catalog_params_optimal.parquet')

   - Each row: one catalog (simulation run).

   - Columns: catalog_number and 6 cosmological/astrophysical parameters.

   - Shape: (1000, 7)

   - File format: Parquet (gzip compressed)



Data Dictionary

----------------------------------------------------------------------
Column             Physical Meaning                         Units               
----------------------------------------------------------------------
M_g                Gas mass (including circumgalactic medium) Msun                
M_star             Stellar mass                             Msun                
M_BH               Black hole mass                          Msun                
M_t                Total mass (DM + gas + stars + BH)       Msun                
V_max              Maximum circular velocity                km/s                
sigma_v            Velocity dispersion (all particles)      km/s                
Z_g                Mass-weighted gas metallicity            Zsun                
Z_star             Mass-weighted stellar metallicity        Zsun                
SFR                Star formation rate                      Msun/yr             
J_spin             Subhalo spin modulus                     kpc km/s            
V_pecu             Subhalo peculiar velocity modulus        km/s                
R_star             Half-mass radius (stars)                 kpc                 
R_t                Half-mass radius (total)                 kpc                 
R_max              Radius at V_max                          kpc                 
U                  U-band magnitude                         mag                 
K                  K-band magnitude                         mag                 
g                  g-band magnitude                         mag                 
catalog_number     Catalog index (simulation run)           integer             
Omega_m            Matter density parameter                 dimensionless       
sigma_8            Power spectrum normalization             dimensionless       
A_SN1              SN wind energy per SFR                   dimensionless       
A_SN2              SN wind speed                            dimensionless       
A_AGN1             AGN feedback energy per accretion        dimensionless       
A_AGN2             AGN kinetic mode ejection speed          dimensionless       
----------------------------------------------------------------------

There a no missing values.

Example Usage: Loading and Analyzing the DataFrames
Load the full galaxy DataFrame:
df_galaxies = pd.read_parquet(root+'/galaxies_full_optimal.parquet')

Load the catalog-level DataFrame:
df_catalogs = pd.read_parquet(root+'/catalog_params_optimal.parquet')


Quantitative Summary of the Dataset

Dataset Structure and Size:

- **Galaxy-level DataFrame**: Contains 720,548 galaxies at \( z=0 \), each associated with one of 1,000 simulated catalogs. Each row includes 17 galaxy properties, the catalog index, and 6 cosmological/feedback parameters, for a total of 24 columns.
- **Catalog-level DataFrame**: Contains 1,000 rows (one per simulation run), each with the catalog index and the 6 cosmological/feedback parameters.

All columns are numerical, with no missing values. The galaxy-level data types are predominantly `float64`, with catalog parameters as `float32` or `float64`.

Parameter Ranges and Distributions:

- **Galaxy Properties**: 
  - Stellar mass (\( M_\mathrm{star} \)) ranges from \( 6.4 \times 10^7 \) to \( 3.2 \times 10^{12} \) \( M_\odot \), with a mean of \( 9.2 \times 10^9 \) \( M_\odot \).
  - Black hole mass (\( M_\mathrm{BH} \)) spans 0 to \( 1.9 \times 10^{10} \) \( M_\odot \).
  - Star formation rate (SFR) ranges from 0 to \( 975 \) \( M_\odot/\mathrm{yr} \), with a mean of 0.43 \( M_\odot/\mathrm{yr} \).
  - Other properties (e.g., gas mass, metallicities, kinematic and structural parameters) cover broad, physically plausible ranges.

- **Catalog Parameters**:
  - Matter density (\( \Omega_m \)): 0.10–0.50 (mean 0.30, std 0.12)
  - Power spectrum normalization (\( \sigma_8 \)): 0.60–1.00 (mean 0.80, std 0.12)
  - Feedback parameters:
    - SN wind energy per SFR (\( A_\mathrm{SN1} \)): 0.25–3.99 (mean 1.35, std 1.02)
    - SN wind speed (\( A_\mathrm{SN2} \)): 0.25–3.99 (mean 1.35, std 1.02)
    - AGN feedback energy per accretion (\( A_\mathrm{AGN1} \)): 0.50–2.00 (mean 1.08, std 0.43)
    - AGN kinetic mode ejection speed (\( A_\mathrm{AGN2} \)): 0.50–2.00 (mean 1.08, std 0.43)

All 1,000 catalogs are unique and span the full parameter space.

Galaxy Population and Binning:

- **Galaxies per Catalog**: The number of galaxies per catalog varies widely (e.g., from ~100 to over 1,500), with a broad, non-uniform distribution.
- **Stellar Mass Bins**:
  - Low (\( <10^9 M_\odot \)): 323,240 galaxies
  - Intermediate (\( 10^9–10^{10} M_\odot \)): 269,450 galaxies
  - High (\( >10^{10} M_\odot \)): 127,720 galaxies
- **SFR Bins**:
  - Low (\( <0.1 M_\odot/\mathrm{yr} \)): 420,290 galaxies
  - Intermediate (\( 0.1–1 M_\odot/\mathrm{yr} \)): 227,510 galaxies
  - High (\( >1 M_\odot/\mathrm{yr} \)): 72,750 galaxies

Mean properties (e.g., gas mass, metallicity, kinematics) increase systematically with both stellar mass and SFR.

Correlations and Relationships:

- **Strong Correlations**:
  - Stellar mass is strongly correlated with black hole mass (\( r=0.86 \)), total mass (\( r=0.76 \)), and maximum circular velocity (\( r=0.68 \)).
  - Black hole mass is also strongly correlated with total mass (\( r=0.84 \)).
  - Kinematic properties (e.g., \( V_\mathrm{max} \), \( \sigma_v \)) are highly correlated (\( r=0.96 \)).
  - Photometric bands (U, K, g) are very strongly correlated with each other (\( r>0.96 \)), as expected.

- **Moderate/Weak Correlations**:
  - SFR shows only weak correlation with stellar mass (\( r=0.10 \)), indicating a wide diversity of star formation activity at fixed mass.
  - Gas and stellar metallicities are only weakly correlated with mass and SFR, but show moderate correlation with each other.

- **Scatter Plots**:
  - The stellar mass–black hole mass relation shows a clear positive trend with significant scatter, consistent with observed scaling relations.
  - The stellar mass–SFR relation is broad, reflecting the diversity of galaxy types (quiescent and star-forming).

Feedback Parameters and Galaxy Properties

- The feedback parameters (\( A_\mathrm{SN1} \), \( A_\mathrm{SN2} \), \( A_\mathrm{AGN1} \), \( A_\mathrm{AGN2} \)) are (log)-uniformly and broadly sampled across the 1,000 catalogs, ensuring good coverage of the parameter space.
- The summary statistics by galaxy bins indicate that higher feedback parameters are associated with catalogs containing more massive and more actively star-forming galaxies, but the direct relationship is not immediately apparent from the summary statistics alone. Further, more targeted analysis (e.g., regression or partial correlation) would be required to disentangle the effects of feedback from those of mass and environment.
- The wide range and distribution of feedback parameters, together with the large sample size, make the dataset well-suited for investigating the impact of feedback on galaxy scaling relations, star formation quenching, and black hole growth.

Suitability and Richness for Research

- The dataset is exceptionally rich, with a very large number of galaxies spanning a wide range of physical properties and environments, and with each galaxy linked to a well-sampled set of cosmological and feedback parameters.
- The structure allows for both galaxy-level and catalog-level analyses, including the study of how global parameters affect galaxy populations.
- The broad and uniform sampling of feedback parameters is particularly valuable for constraining the role of feedback in galaxy evolution.
- The absence of missing data, the large dynamic range, and the detailed binning/statistics make this dataset highly suitable for a wide range of research questions in galaxy formation and feedback physics.


**In summary:**  
This dataset provides a comprehensive, high-quality resource for exploring the interplay between cosmological/feedback parameters and galaxy properties, with sufficient statistical power and parameter coverage to robustly address key questions in galaxy evolution.

**Computing Constraint:**
This project analysis is carried out on a MacBook Pro with 8 available CPUs.
Each individual calculation should not be expected to run for more than O(10minutes).