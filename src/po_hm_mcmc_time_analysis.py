import sys
import itertools
import random
import numpy as np
from typing import List, Dict, Any
from scipy.stats import beta, gamma
import time

# Adjust this path based on your directory structure
sys.path.append('../src/utils')

from po_fun import BasicUtils, StatisticalUtils
from po_accelerator import LogLikelihoodCache

def mcmc_partial_order(
    observed_orders: List[List[int]],
    choice_sets: List[List[int]],
    num_iterations: int,
    K: int,
    # Update for rho:
    dr: float,      # multiplicative step size for rho
    drrt: float,    # multiplicative step size for rho & tau (not used in this snippet)
    # Parameter controlling random-walk for mallow_theta:
    sigma_mallow: float,
    noise_option: str,
    mcmc_pt: List[float],
    # Prior hyperparameters:
    rho_prior,
    noise_beta_prior: float,
    mallow_ua: float
) -> Dict[str, Any]:
    """
    Perform MCMC sampling to infer the partial order h, plus parameters (rho, prob_noise, mallow_theta).
    We measure average time for each update category (rho, noise, or U):
      - proposal time,
      - prior time,
      - likelihood time,
      - total time for that category’s iteration.

    Returns a dictionary of the MCMC traces plus the category timing results.
    """

    # ----------------------
    # 0) Setup & Initialization
    # ----------------------
    items = sorted(set(itertools.chain.from_iterable(choice_sets)))
    n = len(items)
    item_to_index = {item: idx for idx, item in enumerate(items)}
    index_to_item = {idx: item for item, idx in item_to_index.items()}

    observed_orders_idx = [
        [item_to_index[it] for it in order]
        for order in observed_orders
    ]

    # Initialize latent matrix Z (n x K)
    Z = np.zeros((n, K), dtype=float)
    h_Z = BasicUtils.generate_partial_order(Z)

    # Initialize parameters
    rho = StatisticalUtils.rRprior(rho_prior)
    prob_noise = StatisticalUtils.rPprior(noise_beta_prior)
    mallow_theta = StatisticalUtils.rTprior(mallow_ua)

    log_likelihood_cache = LogLikelihoodCache()

    # ----------------------
    # 1) MCMC storage
    # ----------------------
    Z_trace = []
    h_trace = []
    rho_trace = []
    prob_noise_trace = []
    mallow_theta_trace = []

    proposed_rho_vals = []
    proposed_prob_noise_vals = []
    proposed_mallow_theta_vals = []
    proposed_Zs = []

    acceptance_decisions = []
    acceptance_rates = []
    log_likelihood_currents = []
    log_likelihood_primes = []
    num_acceptances = 0

    # 2) Timing accumulators
    # We'll store: 
    # times_data["rho"] = { "count":0, "proposal":0., "prior":0., "likelihood":0., "total":0. }
    # similarly for "noise" and "u".
    times_data = {
        "rho":    {"count":0, "proposal":0.0, "prior":0.0, "likelihood":0.0, "total":0.0},
        "noise":  {"count":0, "proposal":0.0, "prior":0.0, "likelihood":0.0, "total":0.0},
        "u":      {"count":0, "proposal":0.0, "prior":0.0, "likelihood":0.0, "total":0.0},
    }

    # Intervals for progress updates
    progress_intervals = [int(num_iterations * frac) for frac in np.linspace(0.1, 1.0, 10)]

    # Unpack update probabilities
    rho_pct, noise_pct, U_pct = mcmc_pt

    # ----------------------
    # 3) Main MCMC loop
    # ----------------------
    for iteration in range(1, num_iterations + 1):
        r = random.random()

        # Decide update category
        if r < rho_pct:
            category = "rho"
        elif r < (rho_pct + noise_pct):
            category = "noise"
        else:
            category = "u"

        cat_start = time.time()  # overall start for this category’s update

        # ================ A) Update Rho ================
        if category == "rho":
            times_data["rho"]["count"] += 1

            # 1) measure PROPOSAL time
            proposal_start = time.time()
            delta = random.uniform(dr, 1.0 / dr)
            rho_prime = 1.0 - (1.0 - rho)*delta
            if not (0.0 < rho_prime < 1.0):
                rho_prime = rho
            proposal_end = time.time()
            times_data["rho"]["proposal"] += (proposal_end - proposal_start)

            # 2) measure PRIOR time
            prior_start = time.time()
            log_prior_current = (StatisticalUtils.dRprior(rho, rho_prior)
                                 + StatisticalUtils.log_U_prior(Z, rho, K))
            log_prior_proposed = (StatisticalUtils.dRprior(rho_prime, rho_prior)
                                  + StatisticalUtils.log_U_prior(Z, rho_prime, K))
            prior_end = time.time()
            times_data["rho"]["prior"] += (prior_end - prior_start)

            # 3) measure LIKELIHOOD time
            llk_start = time.time()
            # log-likelihood for current
            llk_current = log_likelihood_cache.calculate_log_likelihood(
                Z, h_Z, observed_orders_idx, choice_sets, item_to_index,
                prob_noise, mallow_theta, noise_option
            )
            # Proposed llk is the same because Z isn't changed
            llk_proposed = llk_current
            llk_end = time.time()
            times_data["rho"]["likelihood"] += (llk_end - llk_start)

            log_accept_ratio = (log_prior_proposed + llk_proposed) \
                               - (log_prior_current + llk_current) \
                               - np.log(delta)

            acceptance_probability = min(1.0, np.exp(log_accept_ratio))
            if random.random() < acceptance_probability:
                rho = rho_prime
                num_acceptances += 1
                acceptance_decisions.append(1)
            else:
                acceptance_decisions.append(0)

            proposed_rho_vals.append(rho_prime)
            log_likelihood_currents.append(llk_current)
            log_likelihood_primes.append(llk_proposed)

        # ================ B) Update Noise ================
        elif category == "noise":
            times_data["noise"]["count"] += 1

            proposal_start = time.time()
            # "Proposal" can be reading from the prior or a random-walk approach
            # We'll do it inside each branch
            if noise_option == "mallows_noise":
                epsilon = np.random.normal(0, 1)
                mallow_theta_prime = mallow_theta * np.exp(sigma_mallow * epsilon)
            elif noise_option == "queue_jump":
                prob_noise_prime = StatisticalUtils.rPprior(noise_beta_prior)
            else:
                # fallback
                mallow_theta_prime = mallow_theta
                prob_noise_prime = prob_noise
            proposal_end = time.time()
            times_data["noise"]["proposal"] += (proposal_end - proposal_start)

            # Prior
            prior_start = time.time()
            if noise_option == "mallows_noise":
                log_prior_current = StatisticalUtils.dTprior(mallow_theta, ua=mallow_ua)
                log_prior_proposed = StatisticalUtils.dTprior(mallow_theta_prime, ua=mallow_ua)
            else:  # queue_jump
                log_prior_current = StatisticalUtils.dPprior(prob_noise, noise_beta_prior)
                log_prior_proposed = StatisticalUtils.dPprior(prob_noise_prime, noise_beta_prior)
            prior_end = time.time()
            times_data["noise"]["prior"] += (prior_end - prior_start)

            # Likelihood
            llk_start = time.time()
            llk_current = log_likelihood_cache.calculate_log_likelihood(
                Z, h_Z, observed_orders_idx, choice_sets, item_to_index,
                prob_noise, mallow_theta, noise_option
            )
            if noise_option == "mallows_noise":
                llk_proposed = log_likelihood_cache.calculate_log_likelihood(
                    Z, h_Z, observed_orders_idx, choice_sets, item_to_index,
                    prob_noise, mallow_theta_prime, noise_option
                )
            else:  # queue_jump
                llk_proposed = log_likelihood_cache.calculate_log_likelihood(
                    Z, h_Z, observed_orders_idx, choice_sets, item_to_index,
                    prob_noise_prime, mallow_theta, noise_option
                )
            llk_end = time.time()
            times_data["noise"]["likelihood"] += (llk_end - llk_start)

            if noise_option == "mallows_noise":
                # add Jacobian
                log_accept = ((log_prior_proposed + llk_proposed)
                              - (log_prior_current + llk_current)
                              + np.log(mallow_theta / mallow_theta_prime))
            else:  # queue_jump
                log_accept = ((log_prior_proposed + llk_proposed)
                              - (log_prior_current + llk_current))

            accept_prob = min(1.0, np.exp(log_accept))
            if random.random() < accept_prob:
                if noise_option == "mallows_noise":
                    mallow_theta = mallow_theta_prime
                else:
                    prob_noise = prob_noise_prime
                num_acceptances += 1
                acceptance_decisions.append(1)
            else:
                acceptance_decisions.append(0)

            if noise_option == "mallows_noise":
                proposed_mallow_theta_vals.append(mallow_theta_prime)
            else:
                proposed_prob_noise_vals.append(prob_noise_prime)

            log_likelihood_currents.append(llk_current)
            log_likelihood_primes.append(llk_proposed)

        # ================ C) Update U  ================
        else:
            times_data["u"]["count"] += 1

            # Proposal
            proposal_start = time.time()
            i = random.randint(0, n - 1)
            current_row = Z[i, :].copy()
            Sigma = rho * np.eye(K)
            proposed_row = np.random.multivariate_normal(current_row, Sigma)
            Z_prime = Z.copy()
            Z_prime[i, :] = proposed_row
            proposal_end = time.time()
            times_data["u"]["proposal"] += (proposal_end - proposal_start)

            # Build partial order
            # There's no separate "prior" on the row alone, 
            # but we do measure 'prior' as the time to compute log_U_prior
            prior_start = time.time()
            h_Z_prime = BasicUtils.generate_partial_order(Z_prime)
            lp_cur = StatisticalUtils.log_U_prior(Z, rho, K)
            lp_prop = StatisticalUtils.log_U_prior(Z_prime, rho, K)
            prior_end = time.time()
            times_data["u"]["prior"] += (prior_end - prior_start)

            # Likelihood
            llk_start = time.time()
            llk_current = log_likelihood_cache.calculate_log_likelihood(
                Z, h_Z, observed_orders_idx, choice_sets, item_to_index,
                prob_noise, mallow_theta, noise_option
            )
            llk_proposed = log_likelihood_cache.calculate_log_likelihood(
                Z_prime, h_Z_prime, observed_orders_idx, choice_sets, item_to_index,
                prob_noise, mallow_theta, noise_option
            )
            llk_end = time.time()
            times_data["u"]["likelihood"] += (llk_end - llk_start)

            log_accept = (lp_prop + llk_proposed) - (lp_cur + llk_current)
            accept_prob = min(1.0, np.exp(log_accept))
            if random.random() < accept_prob:
                Z = Z_prime
                h_Z = h_Z_prime
                num_acceptances += 1
                acceptance_decisions.append(1)
            else:
                acceptance_decisions.append(0)
            proposed_Zs.append(Z_prime)

            log_likelihood_currents.append(llk_current)
            log_likelihood_primes.append(llk_proposed)

        # 4) End of this category’s update => measure total
        cat_end = time.time()
        times_data[category]["total"] += (cat_end - cat_start)

        # 5) Store current state
        Z_trace.append(Z.copy())
        h_trace.append(h_Z.copy())
        rho_trace.append(rho)
        prob_noise_trace.append(prob_noise)
        mallow_theta_trace.append(mallow_theta)

        acceptance_rates.append(num_acceptances / iteration if iteration else 0.0)

        # Possibly print progress
        if iteration in progress_intervals:
            print(f"Iteration {iteration}/{num_iterations} - Acceptance Rate: {acceptance_rates[-1]:.2%}")

    overall_acceptance_rate = num_acceptances / (num_iterations or 1)
    print(f"\nOverall Acceptance Rate after {num_iterations} iterations: {overall_acceptance_rate:.2%}")

    # ----------------------------
    # Summarize average times
    # ----------------------------
    # We'll compute for each category the average of (proposal, prior, likelihood, total).
        # ignoring any category that had zero updates (count=0).
    # Compute average times per category (ignoring categories with zero updates)
    avg_times = {}
    for cat in ["rho", "noise", "u"]:
        count = times_data[cat]["count"]
        if count > 0:
            avg_times[cat] = {
                "proposal_avg":   times_data[cat]["proposal"]   / count,
                "prior_avg":      times_data[cat]["prior"]      / count,
                "likelihood_avg": times_data[cat]["likelihood"] / count,
                "total_avg":      times_data[cat]["total"]      / count
            }
        else:
            avg_times[cat] = {
                "proposal_avg":   0.0,
                "prior_avg":      0.0,
                "likelihood_avg": 0.0,
                "total_avg":      0.0
            }

    # Compute overall ("all") averages by summing over all categories
    total_count = 0
    total_proposal = 0.0
    total_prior = 0.0
    total_likelihood = 0.0
    total_total = 0.0

    for cat in ["rho", "noise", "u"]:
        total_count += times_data[cat]["count"]
        total_proposal += times_data[cat]["proposal"]
        total_prior += times_data[cat]["prior"]
        total_likelihood += times_data[cat]["likelihood"]
        total_total += times_data[cat]["total"]

    if total_count > 0:
        overall_avg = {
            "proposal_avg": total_proposal / total_count,
            "prior_avg": total_prior / total_count,
            "likelihood_avg": total_likelihood / total_count,
            "total_avg": total_total / total_count
        }
    else:
        overall_avg = {
            "proposal_avg": 0.0,
            "prior_avg": 0.0,
            "likelihood_avg": 0.0,
            "total_avg": 0.0
        }

    # Compute the proportion of each component relative to its category total (in percent)
    proportions = {}
    for cat in ["rho", "noise", "u"]:
        total_cat = avg_times[cat]["total_avg"]
        if total_cat > 0:
            proportions[cat] = {
                "proposal_pct": (avg_times[cat]["proposal_avg"] / total_cat) * 100,
                "prior_pct": (avg_times[cat]["prior_avg"] / total_cat) * 100,
                "likelihood_pct": (avg_times[cat]["likelihood_avg"] / total_cat) * 100,
            }
        else:
            proportions[cat] = {
                "proposal_pct": 0,
                "prior_pct": 0,
                "likelihood_pct": 0,
            }

    # Print overall summary
    print("\n--- Overall Average Times (in seconds) ---")
    print(f"Overall count: {total_count}")
    for k, v in overall_avg.items():
        print(f"  {k}: {v:.6f}")

    # Print per-category summary along with proportions
    print("\n--- Average Times (in seconds) Per Update Category ---")
    for cat in ["rho", "noise", "u"]:
        print(f"\nCategory '{cat}': count = {times_data[cat]['count']}")
        for k, v in avg_times[cat].items():
            print(f"  {k}: {v:.6f}")
        print("  Component proportions (as % of total):")
        for k, v in proportions[cat].items():
            print(f"    {k}: {v:.2f}%")
    print("------------------------------------------------------")

    # Return results
    return {
        "Z_trace": Z_trace,
        "h_trace": h_trace,
        "index_to_item": index_to_item,
        "item_to_index": item_to_index,
        "rho_trace": rho_trace,
        "prob_noise_trace": prob_noise_trace,
        "mallow_theta_trace": mallow_theta_trace,
        "proposed_rho_vals": proposed_rho_vals,
        "proposed_prob_noise_vals": proposed_prob_noise_vals,
        "proposed_mallow_theta_vals": proposed_mallow_theta_vals,
        "proposed_Zs": proposed_Zs,
        "acceptance_rates": acceptance_rates,
        "acceptance_decisions": acceptance_decisions,
        "log_likelihood_currents": log_likelihood_currents,
        "log_likelihood_primes": log_likelihood_primes,
        "overall_acceptance_rate": overall_acceptance_rate,
        # Detailed timing accumulation
        "times_data": times_data,   # raw sums + counts
        "avg_times": avg_times      # computed average times
    }

import numpy as np

if __name__ == "__main__":
    """
    Minimal example usage:
    We'll define some dummy data with n=8 items and 4 assessors,
    run the chain for 30 iterations,
    and then output the final average times for each update category (rho, noise, u).
    """
    # Define the number of items and assessors
    n = 8
    assessors = [1, 2, 3, 4]

    # Define dummy observed orders and choice sets for each assessor.
    # Here each list represents an order over a subset (or complete set) of items.
    # You can adjust these orders as needed.
    observed_orders = [
        [0, 3, 1, 4],   # Assessor 1 observed order
        [2, 5, 7, 6],   # Assessor 2 observed order
        [1, 2, 3, 0],   # Assessor 3 observed order
        [7, 6, 5, 4]    # Assessor 4 observed order
    ]
    
    # For simplicity, we set the choice sets to be the same as the orders here,
    # but these could be different (e.g., if assessors only consider a subset of items).
    choice_sets = [
        [0, 1, 3, 4],   # Assessor 1 choice set
        [2, 5, 6, 7],   # Assessor 2 choice set
        [0, 1, 2, 3],   # Assessor 3 choice set
        [4, 5, 6, 7]    # Assessor 4 choice set
    ]

    # Set MCMC parameters
    num_iterations = 30000   # For a minimal example run; adjust to 10000 for full-scale runs.
    K = 2                 # Dimensionality of the latent space
    dr = 1.1
    drrt = 1.05
    sigma_mallow = 0.5
    noise_option = "queue_jump"  
    mcmc_pt = [0.3, 0.2, 0.5]

    rho_prior = 0.1667
    noise_beta_prior = 1.0
    mallow_ua = 2.0

    # Call the MCMC function (make sure mcmc_partial_order is defined/imported)
    results = mcmc_partial_order(
        observed_orders=observed_orders,
        choice_sets=choice_sets,
        num_iterations=num_iterations,
        K=K,
        dr=dr,
        drrt=drrt,
        sigma_mallow=sigma_mallow,
        noise_option=noise_option,
        mcmc_pt=mcmc_pt,
        rho_prior=rho_prior,
        noise_beta_prior=noise_beta_prior,
        mallow_ua=mallow_ua
    )

    # For example, you can access results["avg_times"] to see the final average times
    # for each update category.
    print("\nMCMC Done. See above for average times breakdown.")
