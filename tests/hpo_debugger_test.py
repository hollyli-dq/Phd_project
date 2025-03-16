import os
import yaml
import numpy as np
import sys
import seaborn as sns
import math
from scipy.integrate import quad
import random
import matplotlib.pyplot as plt
from scipy.stats import beta, expon, norm, uniform, kstest



# Ensure your module paths are correct:
sys.path.append('/home/doli/Desktop/research/coding/BayesianPartialOrders/src')  # Adjust if necessary
from hpo_po_hm_mcmc import mcmc_simulation_hpo

from po_fun import BasicUtils, StatisticalUtils,GenerationUtils
sys.path.append('/home/doli/Desktop/research/coding/BayesianPartialOrders/src/utils')  # Example path

# Path to your configuration file and output folder.
yaml_file = "/home/doli/Desktop/research/coding/BayesianPartialOrders/hpo_mcmc_configuration.yaml"
outputfilepath = "/home/doli/Desktop/research/coding/BayesianPartialOrders/tests/hpo_test_output/"

# Example input data.
M0 = [0, 1, 2, 3, 4, 5]
assessors = [1, 2, 3]
M_a_dict = {
    1: [0, 2, 4],
    2: [1, 2, 3, 4, 5],
    3: [1, 2, 3, 4, 5]
}
O_a_i_dict = {
    1: [[0, 2, 4], [0, 2]],
    2: [[1, 2, 5], [1, 3], [1, 4, 5]],
    3: [[1, 3], [1, 3, 4], [2, 4, 5]]
}
observed_orders = None 
alpha = np.array([0.5, -0.2, 0.3, 0.1, 0.0, 1.2]) 


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

config = load_config(yaml_file)

num_iterations = config["mcmc"]["num_iterations_debug"]
K = config["mcmc"]["K"]
dr = config["rho"]["dr"]
drrt=config["rhotau"]["drrt"]
noise_option = config["noise"]["noise_option"]
sigma_mallow = config["noise"]["sigma_mallow"]

prior_config = config["prior"]
rho_prior = prior_config["rho_prior"]
noise_beta_prior = prior_config["noise_beta_prior"]
mallow_ua = prior_config["mallow_ua"] 

rho_tau_update = config["reversible_two_factors"]["rho_tau_update"] 


def check_log_likelihood(results: dict) -> float:
    """
    Check that the sum of the current and proposed log likelihoods is close to 0.
    """
    llk_sum = np.sum(results["log_likelihood_currents"] + results["log_likelihood_primes"])
    print("Sum of log likelihood currents and proposed values:", llk_sum)
    if not np.isclose(llk_sum, 0.0, atol=1e-6):
        print("WARNING: The sum of log likelihood values is not 0!")
    else:
        print("Log likelihood values sum to 0 as expected.")
    return llk_sum

def check_param(samples, label, dist, dist_params, output_filename, tol=1e-4):
    """
    General function to check MCMC chain diagnostics against a theoretical distribution.
    
    If the label is 'rho', the theoretical distribution is assumed to be Beta(1, fac)
    truncated to [0, 1-tol]. For other labels, the full distribution is used.
    
    Parameters:
      samples: numpy array of MCMC samples (1D)
      label: string, name of the parameter (e.g., "rho", "P", or "U_entry")
      dist: a scipy.stats distribution (e.g., beta, norm, or expon)
      dist_params: tuple of parameters for the distribution 
                   (e.g., (1, fac) for Beta, (0, scale) for norm, etc.)
      output_filename: string, file name (without path) to save the histogram plot.
      tol: tolerance for truncation (only used for 'rho').
    """
    if label.lower() == "rho":
        a, b = dist_params
        norm_const = beta.cdf(1 - tol, a, b)
        truncated_pdf = lambda x: beta.pdf(x, a, b) / norm_const
        theoretical_mean, _ = quad(lambda x: x * truncated_pdf(x), 0, 1 - tol)
        theoretical_var, _ = quad(lambda x: (x - theoretical_mean)**2 * truncated_pdf(x), 0, 1 - tol)
        x_min, x_max = 0.5, np.max(samples)
        x_vals = np.linspace(x_min, x_max, 10000)
        pdf_vals = beta.pdf(x_vals, a, b) / norm_const
    else:
        theoretical_mean = dist.mean(*dist_params)
        theoretical_var = dist.var(*dist_params)
        x_min = np.min(samples)
        x_max = np.max(samples)
        x_vals = np.linspace(x_min, x_max, 10000)
        pdf_vals = dist.pdf(x_vals, *dist_params)
    
    sample_mean = np.mean(samples)
    sample_var = np.var(samples)
    
    print(f"--- {label} Diagnostics ---")
    print("Theoretical mean:", theoretical_mean)
    print("Sample mean:", sample_mean)
    print("Theoretical variance:", theoretical_var)
    print("Sample variance:", sample_var)
    
    ks_stat, p_value = kstest(samples, 
                              lambda x: beta.cdf(x, *dist_params) / beta.cdf(1-tol, *dist_params)
                              if label.lower() == "rho" else dist(*dist_params).cdf(x))
    print("KS statistic:", ks_stat)
    print("KS test p-value:", p_value)

    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))
    plt.hist(samples, bins=250, density=True, alpha=0.5, label="MCMC samples")
    plt.plot(x_vals, pdf_vals, "r-", lw=2, label=f"{dist.name.capitalize()} PDF")
    plt.xlabel(label)
    plt.ylabel("Density")
    plt.title(f"Histogram of {label} samples with Theoretical Density")
    plt.legend()
    plt.xlim(x_min, x_max)

    
    # Build full output path from the directory and filename.
    full_output_path = os.path.join(outputfilepath, output_filename)
    plt.savefig(full_output_path)
    print(f"Plot saved to: {full_output_path}")
    plt.show()

def check_rho():
    results = mcmc_simulation_hpo(
        num_iterations=num_iterations,
        M0=M0,
        assessors=assessors,
        M_a_dict=M_a_dict,
        O_a_i_dict=O_a_i_dict,
        observed_orders=observed_orders,
        alpha=alpha,
        K=K,          
        dr=dr,
        drrt=drrt,
        sigma_mallow=sigma_mallow,
        noise_option=noise_option,
        mcmc_pt=[1, 0, 0, 0,0],
        rho_prior=rho_prior,
        noise_beta_prior=noise_beta_prior,
        mallow_ua=mallow_ua,
        rho_tau_update=rho_tau_update,
        random_seed=42
    )
    
    rho_samples = np.array(results["rho_trace"])
    check_log_likelihood(results)
    check_param(rho_samples, "rho", beta, (1, rho_prior), "hpo_rho_hist.png")

def check_P():
    
    results = mcmc_simulation_hpo(
        num_iterations=num_iterations,
        M0=M0,
        assessors=assessors,
        M_a_dict=M_a_dict,
        O_a_i_dict=O_a_i_dict,
        observed_orders=observed_orders,
        alpha=alpha,
        K=K,            
        dr=dr,
        drrt=drrt,
        sigma_mallow=sigma_mallow,
        noise_option=noise_option,
        mcmc_pt=[0, 0, 1, 0,0],
        rho_prior=rho_prior,
        noise_beta_prior=noise_beta_prior,
        mallow_ua=mallow_ua,
        rho_tau_update=rho_tau_update,
        random_seed=123
    )
    
    p_samples = np.array(results["prob_noise_trace"])
    check_log_likelihood(results)
    print(np.mean(results['acceptance_rates']))
    check_param(p_samples, "P", beta, (1, noise_beta_prior), "hpo_p_hist.png")

def check_theta():    
    results = mcmc_simulation_hpo(
        num_iterations=num_iterations,
        M0=M0,
        assessors=assessors,
        M_a_dict=M_a_dict,
        O_a_i_dict=O_a_i_dict,
        observed_orders=observed_orders,
        alpha=alpha,
        K=K,            
        dr=dr,
        drrt=drrt,
        sigma_mallow=sigma_mallow,
        noise_option=noise_option,
        mcmc_pt=[0, 0 , 1, 0,0],
        rho_prior=rho_prior,
        noise_beta_prior=noise_beta_prior,
        mallow_ua=mallow_ua,
        rho_tau_update=rho_tau_update,
        random_seed=123
    )
    
    theta_samples = np.array(results["mallow_theta_trace"])
    check_log_likelihood(results)
    check_param(theta_samples, "theta", expon, (0, 1/mallow_ua), "hpo_theta_hist.png")

def check_tau():
    
    results = mcmc_simulation_hpo(
        num_iterations=num_iterations,
        M0=M0,
        assessors=assessors,
        M_a_dict=M_a_dict,
        O_a_i_dict=O_a_i_dict,
        observed_orders=observed_orders,
        alpha=alpha,
        K=K,            
        dr=dr,
        drrt=drrt,
        sigma_mallow=sigma_mallow,
        noise_option=noise_option,
        mcmc_pt=[0, 1, 0, 0,0],
        rho_prior=rho_prior,
        noise_beta_prior=noise_beta_prior,
        mallow_ua=mallow_ua,
        rho_tau_update=rho_tau_update,
        random_seed=123
    )
    
    tau_samples = np.array(results["tau_trace"])
    check_log_likelihood(results)
    check_param(tau_samples, "tau", uniform, (0, 1), "hpo_tau_hist.png")



def check_rho_tau():    
    results = mcmc_simulation_hpo(
        num_iterations=num_iterations,
        M0=M0,
        assessors=assessors,
        M_a_dict=M_a_dict,
        O_a_i_dict=O_a_i_dict,
        observed_orders=observed_orders,
        alpha=alpha,
        K=K,            
        dr=dr,
        drrt=drrt,
        sigma_mallow=sigma_mallow,
        noise_option=noise_option,
        mcmc_pt=[0, 1, 0, 0,0],
        rho_prior=rho_prior,
        noise_beta_prior=noise_beta_prior,
        mallow_ua=mallow_ua,
        rho_tau_update=True,
        random_seed=123
    )
    
    tau_samples = np.array(results["tau_trace"])
    rho_samples = np.array(results["rho_trace"]) 
    check_log_likelihood(results)
    check_param(tau_samples, "tau", uniform, (0, 1), "hpo_tau_hist_drrt.png")
    check_param(rho_samples, "rho", beta, (1, rho_prior), "hpo_rho_hist_drrt.png")


def check_U0():
    results = mcmc_simulation_hpo(
        num_iterations=num_iterations,
        M0=M0,
        assessors=assessors,
        M_a_dict=M_a_dict,
        O_a_i_dict=O_a_i_dict,
        observed_orders=observed_orders,
        alpha=alpha,
        K=K,            
        dr=dr,
        drrt=drrt,
        sigma_mallow=sigma_mallow,
        noise_option=noise_option,
        mcmc_pt=[0, 0, 0, 1, 0],
        rho_prior=rho_prior,
        noise_beta_prior=noise_beta_prior,
        mallow_ua=mallow_ua,
        rho_tau_update=rho_tau_update,
        random_seed=123
    )

    U0_trace= results["U0_trace"]
    u_samples = np.array([U0[0, 0] for U0 in U0_trace]) 
    check_log_likelihood(results)
    # Compare to N(0, sigma_u)
    check_param(u_samples, "U0", norm, (0, 1), "Hpo_U0_hist.png")



def check_Ua():
    """
    Runs the MCMC simulation via mcmc_simulation_hpo, then performs checks on Ua vs U0:
      - Verifies that Ua[j, k] - tau * U0[j, k] follows Normal(0, (1 - tau^2)*Sigma_rho[k,k])
      - Computes correlation of Ua[j, k] vs U0[j, k] across MCMC iterations
      - Checks that U[a][0,0] is normally distributed across iterations
    """

    # ------------------ 1) Run the MCMC Simulation ------------------ #
    results = mcmc_simulation_hpo(
        num_iterations=num_iterations,
        M0=M0,
        assessors=assessors,
        M_a_dict=M_a_dict,
        O_a_i_dict=O_a_i_dict,
        observed_orders=observed_orders,
        alpha=alpha,
        K=K,
        dr=dr,
        drrt=drrt,
        sigma_mallow=sigma_mallow,
        noise_option=noise_option,
        mcmc_pt=[0,0,0,0,1],
        rho_prior=rho_prior,
        noise_beta_prior=noise_beta_prior,
        mallow_ua=mallow_ua,
        rho_tau_update=rho_tau_update,
        random_seed=42
    )

    Ua_trace = results["Ua_trace"]  # list of length num_iterations; each is dict: assessor -> (n_a, K)
    U0_trace = results["U0_trace"]  # list of length num_iterations; each is (n_global, K)
    
    rho_arr = results["rho_trace"]
    tau_arr = results["tau_trace"]
    rho_mean = np.mean(rho_arr)
    tau_mean = np.mean(tau_arr)

    # Build covariance using mean rho
    cov = BasicUtils.build_Sigma_rho(K, rho_mean)

    # Helper function: return the std dev for dimension k_dim
    def get_std_dev_for_dim(k_dim):
        return math.sqrt((1.0 - tau_mean**2) * cov[k_dim, k_dim])

    # ------------------ 2) Check Differences + Correlations (One Figure per Assessor) ------------------ #
    for a in assessors:
        # M_a_dict[a] = [global indices that assessor a rates]
        if not M_a_dict[a]:  # skip if empty
            continue

        # We’ll store all (j_global, k_dim) pairs in subplots
        item_dim_pairs = []
        diffs_dict = {}  # keyed by (j_global, k_dim), value = list of diffs
        corr_dict = {}   # correlation values

        for j_global in M_a_dict[a]:
            for k_dim in range(K):
                diffs = []
                ua_vals = []
                u0_vals = []

                for it in range(num_iterations):
                    Ua_it_a = Ua_trace[it][a]  # shape (n_a, K)
                    U0_it   = U0_trace[it]    # shape (n_global, K)

                    i_loc = M_a_dict[a].index(j_global)
                    val_u_a = Ua_it_a[i_loc, k_dim]
                    val_u_0 = U0_it[j_global, k_dim]

                    diffs.append(val_u_a - tau_mean * val_u_0)
                    ua_vals.append(val_u_a)
                    u0_vals.append(val_u_0)

                item_dim_pairs.append((j_global, k_dim))
                diffs_dict[(j_global, k_dim)] = diffs

                if len(ua_vals) > 1:
                    corr = np.corrcoef(ua_vals, u0_vals)[0, 1]
                else:
                    corr = float('nan')
                corr_dict[(j_global, k_dim)] = corr

        # Create a figure for this assessor’s differences
        sns.set_style("whitegrid")
        n_plots = len(item_dim_pairs)
        # layout: e.g. 3 columns. Adjust as you see fit
        ncols = 3
        nrows = (n_plots + ncols - 1) // ncols  # ceiling division

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
        axes = np.array(axes).reshape(nrows, ncols)  # ensure 2D array

        for idx, (j_global, k_dim) in enumerate(item_dim_pairs):
            row_idx = idx // ncols
            col_idx = idx % ncols
            ax = axes[row_idx, col_idx]

            diffs = diffs_dict[(j_global, k_dim)]
            std_dev = get_std_dev_for_dim(k_dim)

            # Theoretical distribution: Normal(0, std_dev^2)
            x_min, x_max = min(diffs), max(diffs)
            x_vals = np.linspace(x_min, x_max, 200)
            pdf_vals = norm.pdf(x_vals, 0, std_dev)

            # Plot histogram + theoretical PDF
            ax.hist(diffs, bins=40, density=True, alpha=0.6, label="Diffs")
            ax.plot(x_vals, pdf_vals, 'r-', lw=2, label="Normal PDF")

            # KS test
            ks_stat, p_value = kstest(diffs, "norm", args=(0, std_dev))
            corr_val = corr_dict[(j_global, k_dim)]
            ax.set_title(
                f"Item {j_global}, Dim {k_dim}\n"
                f"Corr={corr_val:.2f}, KS p={p_value:.2e}"
            )
            ax.legend()

        # Remove any empty subplots if they exist
        for extra_idx in range(idx+1, nrows*ncols):
            row_idx = extra_idx // ncols
            col_idx = extra_idx % ncols
            axes[row_idx, col_idx].axis("off")

        plt.suptitle(f"Difference Checks for Assessor {a}\nUa[j,k] - tau*U0[j,k]", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave room for suptitle

        # Save figure
        diff_fig_filename = f"diffs_assessor_{a}.png"
        diff_output_path = os.path.join(outputfilepath, diff_fig_filename)
        plt.savefig(diff_output_path, dpi=150)
        print(f"[INFO] Saved difference plots for assessor {a}: {diff_output_path}")
        plt.show()

    # ------------------ 3) Check Normality of U[a][0,0] across iterations ------------------ #
    for a in assessors:
        if len(M_a_dict[a]) == 0:
            continue
        j_global_0 = M_a_dict[a][0]  # the first object
        ua_00_vals = []
        u0_00_vals = []

        for it in range(num_iterations):
            Ua_it_a = Ua_trace[it][a]
            U0_it   = U0_trace[it]
            if Ua_it_a.shape[0] > 0:
                ua_00_vals.append(Ua_it_a[0, 0])
            u0_00_vals.append(U0_it[j_global_0, 0])

        mean_guess = tau_mean * np.mean(u0_00_vals)
        std_guess = math.sqrt((1.0 - tau_mean**2) * cov[0, 0])

        # Now plot the distribution of U[a][0,0]
        plt.figure(figsize=(6, 4))
        sns.set_style("whitegrid")

        x_min, x_max = min(ua_00_vals), max(ua_00_vals)
        x_vals = np.linspace(x_min, x_max, 200)
        pdf_vals = norm.pdf(x_vals, loc=mean_guess, scale=std_guess)

        plt.hist(ua_00_vals, bins=40, density=True, alpha=0.6, label="U[a][0,0] Samples")
        plt.plot(x_vals, pdf_vals, "r-", lw=2, label="Normal PDF")

        # KS test
        ks_stat, p_value = kstest(ua_00_vals, "norm", args=(mean_guess, std_guess))

        plt.title(
            f"Normality Check: U[a={a}][0,0]\n"
            f"Mean ~ {mean_guess:.2f}, Std ~ {std_guess:.2f}\n"
        )
        plt.legend()

        normal_fig_filename = f"ua_{a}_0_0.png"
        normal_output_path = os.path.join(outputfilepath, normal_fig_filename)
        plt.tight_layout()
        plt.savefig(normal_output_path, dpi=150)
        print(f"[INFO] Saved U[a={a}][0,0] normality plot: {normal_output_path}")
        plt.show()




if __name__ == "__main__":
    param = sys.argv[1].lower()
    if param == "rho":
        check_rho()
    elif param == "p":
        check_P()
    elif param == "theta":
        check_theta()
    elif param == "tau":
        check_tau()
    elif param == "U0":
        check_U0()
    elif param == "Ua":
        check_Ua()
    elif param == "rhotau":
        check_rho_tau()
    else:
        print("Unknown parameter. Please choose one of: rho, P, theta, tau, U.")
