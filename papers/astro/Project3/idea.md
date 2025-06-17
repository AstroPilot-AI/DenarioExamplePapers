
Compare 1D CNN and FCNN neural net architectures for non-linear matter power spectrum emulators in LCDM.
Input parameters include the cosmological parameters as well as the redshift at which the power spectrum is requested.
We want to determine the architecture that is both as accurate as possible and as fast as possible to compute.

Constraints:
- use classy_sz for computing P(k,z) to generate the data.
- Cosmological parameters are omega_b, omega_cdm, H0, logA, n_s. 
- Range should be: 
logA: 2.5 3.5
omega_cdm: 0.08 0.20
omega_b: 0.01933 0.02533
H0 [km/s/Mpc]: 40 100
ns: 0.8 1.2
- The k-range of interest is kmin = 1e-4 [1/Mpc] to kmax = 10 [1/Mpc].
- The z-range of interest is zmin = 0 to zmax = 3.
- P(k) at z should always be plotted/processed in log-log scale.
- Training set should have at least 50k spectra.
- The training/testing data should be created in a different step as training the neural nets.
- The step to train the neural nets should only be for training. Plotting and performances should be done in another step.
