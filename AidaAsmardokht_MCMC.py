
#Aida Asmardokht

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

#loading the data file

os.chdir('C:/Users/Lenovo/Desktop/MCMC')
data = np.loadtxt('mcmc_cosmo_data.txt', skiprows=1)
z_data, dL_data, dL_err = data[:, 0], data[:, 1], data[:, 2]

print(f" Loaded {len(z_data)} data points")

# Convert to PyTorch tensors
z_tensor = torch.tensor(z_data, dtype=torch.float32)
dL_tensor = torch.tensor(dL_data, dtype=torch.float32)
dL_err_tensor = torch.tensor(dL_err, dtype=torch.float32)

# Luminosity distance function
def luminosity_distance(z, H0, Omega_m):
    #d_L(z) = c(1+z)/H0 * ∫[0,z] dz'/sqrt(Ω_m(1+z')^3 + (1-Ω_m))
    c = 3e5  # speed of light in km/s
    
    # Create integration points
    z_points = torch.linspace(0, z, 100)
    integrand = 1.0 / torch.sqrt(Omega_m * (1 + z_points)**3 + (1 - Omega_m))
    integral = torch.trapz(integrand, z_points)
    
    return c * (1 + z) * integral / H0

###########################################################################################################################
###########################################################################################################################



# Gaussian likelihood function
def log_likelihood(H0, Omega_m, z_data, dL_data, dL_err):
    log_likelihood_val = 0.0
   
    # Calculate for each data point
    for i in range(len(z_data)):
        z = z_data[i]
        dL_observed = dL_data[i]
        sigma = dL_err[i]
        
        # Model prediction
        dL_model = luminosity_distance(z, H0, Omega_m)
        
        # Gaussian log-likelihood for this point
        chi2 = ((dL_observed - dL_model) / sigma) ** 2
        log_likelihood_val += -0.5 * chi2 - 0.5 * np.log(2 * np.pi * sigma**2)
        
    return log_likelihood_val
 
############################################################################################################################
########################################################################################################################   
    


# Prior distribution function
def log_prior(H0, Omega_m):
    #Uniform prior distributions:
    #H0 ~ Uniform(50, 100) km/s/Mpc
    #Ω_m ~ Uniform(0.1, 0.9)
    if 50.0 <= H0 <= 100.0 and 0.1 <= Omega_m <= 0.9:
        return 0.0 
    else:
        return -torch.inf

# Posterior distribution function  
def log_posterior(H0, Omega_m, z_data, dL_data, dL_err):
    """
    Log posterior: log P(H0, Ω_m | data) ∝ log P(data | H0, Ω_m) + log P(H0, Ω_m)
    """
    prior = log_prior(H0, Omega_m)
    if not torch.isfinite(torch.tensor(prior)):
        return -torch.inf
    
    likelihood = log_likelihood(H0, Omega_m, z_data, dL_data, dL_err)
    return likelihood + prior
    
#########################################################################################################################
#########################################################################################################################



# Metropolis-Hastings MCMC algorithm
def metropolis_hastings(z_data, dL_data, dL_err, n_steps=5000):
    """
    Metropolis-Hastings MCMC for cosmological parameters
    """
    # Initialize parameters
    H0_current = torch.tensor(70.0)  # Starting value for H0
    Omega_m_current = torch.tensor(0.3)  # Starting value for Ωm
    
    # Store samples
    samples = []
    
    for step in range(n_steps):
        # Gaussian proposal jumps
        H0_proposal = H0_current + torch.normal(0.0, 2.0, size=(1,))
        Omega_m_proposal = Omega_m_current + torch.normal(0.0, 0.05, size=(1,))
        
        # Calculate current and proposed posteriors
        log_post_current = log_posterior(H0_current, Omega_m_current, z_data, dL_data, dL_err)
        log_post_proposal = log_posterior(H0_proposal, Omega_m_proposal, z_data, dL_data, dL_err)
        
        # Acceptance ratio
        acceptance_ratio = torch.exp(log_post_proposal - log_post_current)
        
        # Accept or reject
        if torch.rand(1) < acceptance_ratio:
            H0_current = H0_proposal
            Omega_m_current = Omega_m_proposal
        
        samples.append([H0_current.item(), Omega_m_current.item()])
    
    return torch.tensor(samples)

##############################################################################################################################################
#############################################################################################################################################
# Run MCMC

samples = metropolis_hastings(z_data, dL_data, dL_err, n_steps=5000)

# Remove burn-in (first 1000 samples)
burn_in = 1000
samples_clean = samples[burn_in:]

# Calculate results
H0_mean = samples_clean[:, 0].mean().item()
H0_std = samples_clean[:, 0].std().item()
Omega_m_mean = samples_clean[:, 1].mean().item()
Omega_m_std = samples_clean[:, 1].std().item()

print(f"Results:")
print(f"H0 = {H0_mean:.1f} ± {H0_std:.1f} km/s/Mpc")
print(f"Ωm = {Omega_m_mean:.3f} ± {Omega_m_std:.3f}")

#################################################################################################################################################
##################################################################################################################################################


# Plot posterior distributions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(samples_clean[:, 0], bins=30, density=True, alpha=0.7)
plt.axvline(H0_mean, color='red', linestyle='--', label=f'Mean = {H0_mean:.1f}')
plt.xlabel('H0 (km/s/Mpc)')
plt.ylabel('Probability Density')
plt.title('Posterior Distribution of H0')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(samples_clean[:, 1], bins=30, density=True, alpha=0.7, color='orange')
plt.axvline(Omega_m_mean, color='red', linestyle='--', label=f'Mean = {Omega_m_mean:.3f}')
plt.xlabel('Ωm')
plt.ylabel('Probability Density')
plt.title('Posterior Distribution of Ωm')
plt.legend()

plt.tight_layout()
plt.show()

# Credible intervals
H0_68 = np.percentile(samples_clean[:, 0], [16, 84])
Omega_m_68 = np.percentile(samples_clean[:, 1], [16, 84])

print(f"68% Credible Intervals:")
print(f"H0: [{H0_68[0]:.1f}, {H0_68[1]:.1f}] km/s/Mpc")
print(f"Ωm: [{Omega_m_68[0]:.3f}, {Omega_m_68[1]:.3f}]")