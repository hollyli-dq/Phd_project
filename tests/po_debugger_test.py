import yaml
import numpy as np
from scipy.stats import beta, kstest
import sys 
from scipy.integrate import quad
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import expon, kstest, probplot

# Add the path to the utility modules
sys.path.append('../src')  # Ensure this path points to the directory containing your utility modules
from po_hm_mcmc_all import mcmc_partial_order


yaml_file= "/home/doli/Desktop/research/coding/BayesianPartialOrders/mcmc_config.yaml"
def check_log_likelihood(results):
    """
    Check that the sum of log likelihood currents is 0.
    """
    llk_sum = np.sum(results["log_likelihood_currents"]+results["log_likelihood_primes"])
    print("Sum of log likelihood currents:", llk_sum)
    if not np.isclose(llk_sum, 0.0, atol=1e-6):
        print("WARNING: The sum of log likelihood currents is not 0!")
    else:
        print("Log likelihood currents sum to 0 as expected.")
    return llk_sum
def check_param(samples, label, dist, dist_params, output_filename, num_bins=300, tol=1e-4):
    """
    General function to check MCMC chain diagnostics against a theoretical distribution.
    
    If the label is 'rho', the theoretical distribution is assumed to be Beta(1, fac)
    truncated to [0, 1-tol]. For other labels, the full distribution is used.
    
    Parameters:
      samples: numpy array of MCMC samples (1D)
      label: string, name of the parameter (e.g., "rho", "P", or "U_entry")
      dist: a scipy.stats distribution (e.g., beta, norm, or expon)
      dist_params: tuple of parameters for the distribution 
      output_filename: string, file name to save the histogram plot.
      tol: tolerance for truncation (only used for 'rho').
    """
    # For 'rho', we assume the distribution is truncated Beta on [0, 1-tol].
    if label.lower() == "rho":
        a, b = dist_params
        truncation_point = 1 - tol
        norm_const = dist.cdf(truncation_point, a, b)
        
        # Theoretical moments for truncated distribution
        truncated_pdf = lambda x: dist.pdf(x, a, b) / norm_const
        theoretical_mean, _ = quad(lambda x: x * truncated_pdf(x), 0, truncation_point)
        theoretical_var, _ = quad(lambda x: (x - theoretical_mean)**2 * truncated_pdf(x), 
                                  0, truncation_point)
        
        # Bins and range explicitly aligned with truncation
        bin_edges = np.linspace(0, truncation_point, num_bins + 1)
        x_vals = np.linspace(0, truncation_point, 1000)
        pdf_vals = truncated_pdf(x_vals)

    else:
        # For other parameters
        theoretical_mean = dist.mean(*dist_params)
        theoretical_var = dist.var(*dist_params)
        bin_edges = num_bins
        x_vals = np.linspace(np.min(samples), np.max(samples), 1000)
        pdf_vals = dist.pdf(x_vals, *dist_params)

    sample_mean = np.mean(samples)
    sample_var = np.var(samples)
    
    print(f"--- {label} Diagnostics ---")
    print(f"Theoretical vs Sample Mean: {theoretical_mean:.4f} | {sample_mean:.4f}")
    print(f"Theoretical vs Sample Var:  {theoretical_var:.4f} | {sample_var:.4f}")

    # KS test with corrected CDF
    if label.lower() == "rho":
        ks_cdf = lambda x: dist.cdf(x, a, b) / norm_const
    else:
        ks_cdf = lambda x: dist.cdf(x, *dist_params)
        
    ks_stat, p_value = kstest(samples, ks_cdf)
    print(f"KS Stat: {ks_stat:.3f}, p-value: {p_value:.3f}")

    # Main density plot
    plt.figure(figsize=(8, 5))
    plt.hist(samples, bins=bin_edges, density=True, alpha=0.5, 
             label="Samples", edgecolor='black')
    plt.plot(x_vals, pdf_vals, 'r-', lw=2, label=f"Truncated {dist.name}" if label=='rho' else f"{dist.name} PDF")
    plt.xlabel(label)
    plt.ylabel("Density")
    plt.title(f"{label} Distribution Check")
    plt.legend()
    plt.savefig(output_filename)
    plt.show()

    # Special handling for rho
    if label.lower() == "rho":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Enhanced Trace Plot
        ax1.plot(samples, color='#1f77b4', lw=1, alpha=0.8)  # Better color/visibility
        ax1.set_title(f"Trace Plot: {label}", fontsize=12)
        ax1.set_xlabel("Iteration", fontsize=10)
        ax1.set_ylabel(label, fontsize=10)
        ax1.set_ylim(0, 1 - tol)
        ax1.grid(True, alpha=0.3)

        # Enhanced Histogram
        truncation_point = 1 - tol
        bin_edges = np.linspace(0, truncation_point, num_bins + 1)
        
        hist = sns.histplot(
            samples,
            bins=bin_edges,
            kde=True,
            ax=ax2,
            color='#2ca02c',  # More visible color
            edgecolor='black',  # Defined edge color
            linewidth=0.5,      # Edge line thickness
            alpha=0.7,          # Increased opacity
            stat='density'
        )
        
        # Overlay theoretical PDF
        x_vals = np.linspace(0.5, truncation_point, 1000)
        ax2.plot(x_vals, pdf_vals, 'r-', lw=2, label='Theoretical PDF')
        
        # Adjust axis limits and ticks
        ax2.set_xlim(0.5, truncation_point)
        ax2.set_xticks(np.linspace(0.5, truncation_point, 6))  # Force 6 visible ticks
        ax2.set_xticklabels([f"{x:.1f}" for x in np.linspace(0.5, 1, 6)])  # Show 0-1 labels
        
        # Add mean lines
        ax2.axvline(theoretical_mean, color='purple', linestyle='--', 
                   label=f'Theoretical Mean: {theoretical_mean:.3f}')
        ax2.axvline(sample_mean, color='orange', linestyle='--', 
                   label=f'Sample Mean: {sample_mean:.3f}')
        
        # Improve legend and title
        ax2.legend(loc='upper left', frameon=True)
        ax2.set_title(f"Distribution: {label}", fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        subplot_filename = output_filename.replace(".png", "_subplots.png")
        plt.savefig(subplot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved enhanced subplots to {subplot_filename}")

def check_rho():
    # Load configuration.
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    
    num_iterations = config["mcmc"]["num_iterations"]
    K = config["mcmc"]["K"]
    # Set update probabilities so that only rho is updated.
    mcmc_pt = [1, 0, 0]
    dr = config["rho"]["dr"]
    noise_option = config["noise"]["noise_option"]
    sigma_mallow = config["noise"]["sigma_mallow"]
    
    # Prior hyperparameters.
    rho_prior = config["prior"]["rho_prior"]  # for rho, prior is Beta(1, rho_prior)
    noise_beta_prior = config["prior"]["noise_beta_prior"]
    mallow_ua = config["prior"]["mallow_ua"]
    sigma_u = config["prior"]["sigma_u"]
    
    total_orders = []  # No data.
    subsets = []
    
    results = mcmc_partial_order(
        total_orders,
        subsets,
        num_iterations,
        K,
        dr,
        sigma_mallow,
        noise_option,
        mcmc_pt,
        rho_prior,          # Pass rho_prior here.
        noise_beta_prior,
        mallow_ua
    )
    
    # Extract the chain for rho.
    rho_samples = np.array(results["rho_trace"])
    # Check against Beta(1, rho_prior)
    check_log_likelihood(results)
    check_param(rho_samples, "rho", beta, (1, rho_prior), "rho_hist.png")


def check_P():
    # Load configuration.
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    
    num_iterations = config["mcmc"]["num_iterations"]
    K = config["mcmc"]["K"]
    # Set update probabilities so that only the noise parameter (P) is updated.
    mcmc_pt = [0, 1, 0]
    dr = config["rho"]["dr"]
    # Use "queue_jump" noise option to trigger P update.
    noise_option = "queue_jump"
    sigma_mallow = config["noise"]["sigma_mallow"]
    
    # Prior hyperparameters.
    rho_prior = config["prior"]["rho_prior"]
    noise_beta_prior = config["prior"]["noise_beta_prior"]  # For P, prior: Beta(1, noise_beta_prior)
    mallow_ua = config["prior"]["mallow_ua"]
    sigma_u = config["prior"]["sigma_u"]
    total_orders = []
    subsets = []
    
    results = mcmc_partial_order(
        total_orders,
        subsets,
        num_iterations,
        K,
        dr,
        sigma_mallow,
        noise_option,
        mcmc_pt,
        rho_prior,
        noise_beta_prior,
        mallow_ua
    )
    
    # Extract the chain for P.
    p_samples = np.array(results["prob_noise_trace"])
    # Check against Beta(1, noise_beta_prior)
    check_log_likelihood(results)
    check_param(p_samples, "P", beta, (1, noise_beta_prior), "p_hist.png")


def check_theta():
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    
    num_iterations = config["mcmc"]["num_iterations"]
    K = config["mcmc"]["K"]
    # Only update theta (mallows_theta) by setting the update probabilities to [0, 1, 0].
    mcmc_pt = [0, 1, 0]
    dr = config["rho"]["dr"]
    # For theta update, force noise_option to "mallows_noise".
    noise_option = "mallows_noise"
    sigma_mallow = config["noise"]["sigma_mallow"]
    
    rho_prior = config["prior"]["rho_prior"]
    noise_beta_prior = config["prior"]["noise_beta_prior"]
    mallow_ua = config["prior"]["mallow_ua"]
    
    total_orders = []
    subsets = []
    
    results = mcmc_partial_order(
        total_orders,
        subsets,
        num_iterations,
        K,
        dr,
        sigma_mallow,
        noise_option,
        mcmc_pt,
        rho_prior,
        noise_beta_prior
    )
    
    theta_samples = np.array(results["mallow_theta_trace"])
    sum(results["log_likelihood_currents"])
    # Check theta against an Exponential distribution with rate mallow_ua.
    # In scipy.stats.expon, set loc=0 and scale = 1/mallow_ua.
    check_log_likelihood(results)
    check_param(theta_samples, "theta", expon, (0, 1/mallow_ua), "theta_hist.png")


def check_U():
    from scipy.stats import norm, kstest
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    
    num_iterations = 100000
    K = config["mcmc"]["K"]
    mcmc_pt = [0, 0, 1]  # Only update U.
    dr = config["rho"]["dr"]
    noise_option = config["noise"]["noise_option"]
    sigma_mallow = config["noise"]["sigma_mallow"]
    
    # Prior hyperparameters.
    rho_prior = config["prior"]["rho_prior"]
    noise_beta_prior = config["prior"]["noise_beta_prior"]
    mallow_ua = config["prior"]["mallow_ua"]
    sigma_u = config["prior"]["sigma_u"]
    
    # Use dummy data to ensure n > 0.
    total_orders = []  # Dummy observed order so that n = 3.
    subsets = [[1, 2, 3]]       # Dummy choice set.
    
    results = mcmc_partial_order(
        observed_orders=total_orders,
        choice_sets=subsets,
        num_iterations=num_iterations,
        K=K,
        dr=dr,
        sigma_mallow=sigma_mallow,
        noise_option=noise_option,
        mcmc_pt=mcmc_pt,
        rho_prior=rho_prior,
        noise_beta_prior=noise_beta_prior,
        mallow_ua=mallow_ua
    )
    
    # Extract the chain of Z matrices
    Z_trace = results["Z_trace"]
    
    # Flatten every Z to get all U entries across all iterations
    u_samples = np.array([Z[0, 0] for Z in Z_trace]) # we only check one element's distribution, and it should follows N(0, 1) 

    
    # Check log-likelihood sum (should be 0 in a no-data scenario)
    check_log_likelihood(results)
    
    # Compare to N(0, sigma_u)
    check_param(u_samples, "U_entry", norm, (0, 1), "u_entry_hist.png")



if __name__ == "__main__":
    # Use command-line arguments to choose which parameter to test.
    if len(sys.argv) < 2:
        print("Usage: python check_params.py [rho|P|theta|U]")
        sys.exit(1)
    
    param = sys.argv[1].lower()
    if param == "rho":
        check_rho()
        
    elif param == "p":
        check_P()
    elif param == "theta":
        check_theta()
    elif param == "u":
        check_U()
    else:
        print("Unknown parameter. Please choose one of: rho, P, theta, U.")
