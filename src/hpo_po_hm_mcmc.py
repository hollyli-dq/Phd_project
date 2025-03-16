import numpy as np
import math
from scipy.stats import multivariate_normal, norm

import random
import seaborn as sns
import pandas as pd
from collections import Counter, defaultdict
import itertools
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any
import sys as sys
import copy

import itertools
import random
import math
import numpy as np
from scipy.stats import beta, gamma

# Make sure these paths and imports match your local project structure
sys.path.append('/home/doli/Desktop/research/coding/BayesianPartialOrders/src/utils')  # Example path

from po_fun import BasicUtils, StatisticalUtils#
from mallow_function import Mallows
from po_accelerator import HPO_LogLikelihoodCache
from typing import Dict, List



def mcmc_simulation_hpo(
    num_iterations: int,
    # Hierarchy definition
    M0: List[int],
    assessors: List[int],
    M_a_dict:Dict[int, List[List[int]]],
    # Observed data
    O_a_i_dict: Dict[int, List[List[int]]], 
    observed_orders: Dict[int, List[List[int]]],
    # Additional model parameters
    alpha: np.ndarray,       # shape = (|M0|,)
    K: int,                  # dimension of latent space
    dr: float,  # multiplicative step size for rho
    drrt: float,  # multiplicative step size for tau and rho 
    # noise / priors
    sigma_mallow: float,
    noise_option: str,
    # pcts 
    mcmc_pt: List[float],
    # priors
    rho_prior, 
    noise_beta_prior: float,
    mallow_ua: float,
    # Optional
    rho_tau_update: bool = False,
    random_seed: int = 42 
                       ) -> Dict[str, Any]:
    


    # ----------------- Initialization -----------------
    # 0) Seeds
    np.random.seed(random_seed)
    random.seed(random_seed)

    # M0 is given; create an item mapping (assuming items are unique in M0)
    items = sorted(set(M0))
    item_to_index = {item: idx for idx, item in enumerate(items)}


    # 1) Sample initial rho, tau from prior
    rho = StatisticalUtils.rRprior(rho_prior)  # initial rho from its prior
    prob_noise = StatisticalUtils.rPprior(noise_beta_prior) # Beta(1, noise_beta_prior)
    mallow_theta = StatisticalUtils.rTprior(mallow_ua)   # e.g. uniform(0.1,0.9)
    tau = StatisticalUtils.rTauprior()          

    Sigma_rho = BasicUtils.build_Sigma_rho(K,rho)

    # 3) Sample global U^(0) from N(0, Sigma_rho)
    rng = np.random.default_rng(random_seed)
    n_global = len(M0)
    U0 = rng.multivariate_normal(mean=np.zeros(K), cov=Sigma_rho, size=n_global)

    # Initilize of the Ua 

    U_a_dict = {}
    for a in assessors:
        M_a = M_a_dict.get(a, [])
        n_a = len(M_a)
        Ua = np.zeros((n_a, K), dtype=float)
        for i_loc, j_global in enumerate(M_a):
            mean_vec = tau * U0[j_global, :]
            cov_mat=(1.0 - tau**2) * rho * np.eye(K)
            Ua[i_loc, :] = rng.multivariate_normal(mean=mean_vec, cov=cov_mat)
        U_a_dict[a] = Ua


    # Storage for traces.
    U0_trace = []
    Ua_trace = []
    H_trace = []  
    rho_trace = []
    tau_trace = []
    prob_noise_trace = []
    mallow_theta_trace = []
    proposed_rho_vals = []
    proposed_tau_vals = []
    proposed_prob_noise_vals = []
    proposed_mallow_theta_vals = []
    proposed_U0 = []
    proposed_U_a = {}
    acceptance_decisions = []
    acceptance_rates = []
    log_likelihood_currents = []
    log_likelihood_primes = []
    update_records=[]
    

    num_acceptances = 0
    # Unpack update probabilities.
    rho_pct, tau_pct,noise_pct,U0_pct,Ua_pct = mcmc_pt
    thresh_rho = rho_pct
    thresh_tau = rho_pct + tau_pct
    thresh_noise = rho_pct + noise_pct+tau_pct
    threshold_U0_pct = rho_pct + noise_pct+ tau_pct+ U0_pct
    threshold_Ua_pct = rho_pct + noise_pct+tau_pct+ U0_pct+ Ua_pct


    rho_tau_pct = rho_pct+tau_pct
    progress_intervals = [int(num_iterations * frac) for frac in np.linspace(0.1, 1.0, 10)]


    for iteration in range(1,num_iterations+1):
        
        r = random.random()
        # We'll define "U" as a dict holding both global a local latents
        U = {0: U0, **U_a_dict}
        accepted_this_iter = False


        #  Build partial orders h_U from the current latents
        h_U = StatisticalUtils.build_hierarchical_partial_orders(
            M0=M0,
            assessors=assessors,
            M_a_dict=M_a_dict,
            U0=U0,
            U_a_dict=U_a_dict,
            alpha=alpha
            # link_inv=... if not the default

        )

        log_llk_current = HPO_LogLikelihoodCache.calculate_log_likelihood_hpo(
            U=U,
            h_U=h_U,
            observed_orders=observed_orders,
            M_a_dict=M_a_dict,
            O_a_i_dict=O_a_i_dict,
            item_to_index=item_to_index,  # or your real index map if needed
            prob_noise=prob_noise,
            mallow_theta=mallow_theta,
            noise_option=noise_option,
            alpha=alpha
        )

        if r < thresh_rho and rho_tau_update ==False:
            update_category = 0

            
            delta = random.uniform(dr, 1.0 / dr)
            rho_prime = 1.0 - (1.0 - rho) * delta
            if not (0.0 < rho_prime < 1.0):
                rho_prime = rho
            
            Sigma_rho_prime = BasicUtils.build_Sigma_rho(K,rho_prime) 

            log_prior_current = (
                StatisticalUtils.dRprior(rho,rho_prior) 
            )
            log_prior_proposed = StatisticalUtils.dRprior(rho_prime,rho_prior)

            log_llk_proposed = log_llk_current

            log_acceptance_ratio = (log_prior_proposed + log_llk_proposed) - \
                                   (log_prior_current +log_llk_current) -  np.log(delta)

            log_likelihood_currents.append(log_llk_current)
            log_likelihood_primes.append(log_llk_proposed)

            acceptance_probability = min(1.0, np.exp(log_acceptance_ratio))
            if random.random() < acceptance_probability:
                rho = rho_prime
                U0 = U0
                U_a_dict = U_a_dict
                Sigma_rho = Sigma_rho_prime  # for future steps
                log_llk_current = log_llk_proposed
                num_acceptances += 1
                acceptance_decisions.append(1)
                accepted_this_iter = True
            else:
                acceptance_decisions.append(0)
            proposed_rho_vals.append(rho_prime)                        



        # ---- B) Update tau ----

        elif r < thresh_tau and rho_tau_update ==False:
            tau_prime = StatisticalUtils.rTauprior()
            U_a_dict_prime = {}
            update_category = 1 
            for a in assessors:
                M_a = M_a_dict.get(a, [])
                n_a = len(M_a)
                Ua_prime = np.zeros((n_a, K), dtype=float)
                for i_loc, j_global in enumerate(M_a):
                    mean_vec = tau_prime * U0[j_global, :]
                    cov_mat = (1.0 - tau_prime**2) * Sigma_rho
                    Ua_prime[i_loc, :] = rng.multivariate_normal(mean=mean_vec, cov=cov_mat)
                U_a_dict_prime[a] = Ua_prime

            U_prime = {"U0": U0, "U_a_dict": U_a_dict_prime}

            h_U_prime = StatisticalUtils.build_hierarchical_partial_orders(
                M0=M0,
                assessors=assessors,
                M_a_dict=M_a_dict,
                U0=U0,            # U^(0) is unchanged
                U_a_dict=U_a_dict_prime,
                alpha=alpha
            )



            # Evaluate the new log-likelihood
            log_llk_proposed = log_llk_current

               # old prior
            log_prior_current = (

                 StatisticalUtils.log_U_a_prior(U_a_dict, tau, rho, K, M_a_dict, U0)
      
            )
            # new prior
            log_prior_proposed = (
      
                StatisticalUtils.log_U_a_prior(U_a_dict, tau_prime, rho, K, M_a_dict, U0)
            )

            # Data-likelihood may or may not change if code uses tau explicitly in the likelihood
            # We'll assume it does not, or does so in partial
            # We'll skip re-sampling U => same partial approach
            log_accept_ratio = log_prior_proposed - log_prior_current 
            
            log_accept_ratio=min(log_accept_ratio,700)
            accept_prob = min(1.0, math.exp(min(log_accept_ratio, 700)))

            log_likelihood_primes.append(log_llk_proposed)
            log_likelihood_currents.append(log_llk_current)

            if random.random() < accept_prob:
                tau = tau_prime
                num_acceptances += 1
                acceptance_decisions.append(1)
                accepted_this_iter = True

            else:
                acceptance_decisions.append(0)
            proposed_tau_vals.append(tau_prime)



        ## Update rho and tau together
        elif r < rho_tau_pct  and rho_tau_update ==True:
            delta = random.uniform(drrt, 1.0 / drrt)
            rho_prime = 1.0 - (1.0 - rho) * delta
            tau_prime =1.0 - (1.0 - tau) * delta
            if not (0.0 < rho_prime < 1.0):
                rho_prime = rho
            if not (0.0 < tau_prime < 1.0):
                tau_prime = tau
            Sigma_rho_prime = BasicUtils.build_Sigma_rho(K,rho_prime) 

            log_prior_current= (
                StatisticalUtils.log_U_prior(U0, rho, K)
                +StatisticalUtils.log_U_a_prior(U_a_dict, tau, rho, K, M_a_dict, U0)
                + StatisticalUtils.dTauprior(tau)
                + StatisticalUtils.dRprior(rho, rho_prior)
            )
            log_prior_proposed= (
                StatisticalUtils.log_U_prior(U0, rho_prime, K)
                + StatisticalUtils.log_U_a_prior(U_a_dict, tau_prime, rho_prime, K, M_a_dict, U0)
                + StatisticalUtils.dTauprior(tau_prime)
                + StatisticalUtils.dRprior(rho_prime, rho_prior)
            )
            ## The likelihood terms cancels out since there is no change in the input for likelihood calcualtion 
            log_acceptance_ratio = log_prior_proposed- log_prior_current - 2 * math.log(delta)
            if random.random() < min(1.0, np.exp(log_acceptance_ratio)):
                rho = rho_prime
                tau = tau_prime
                num_acceptances += 1
                acceptance_decisions.append(1)

            else:
                acceptance_decisions.append(0)
            proposed_tau_vals.append(tau_prime)
            proposed_rho_vals.append(rho_prime)   

        # ---- B) Update noise parameter ----
        elif r < thresh_noise:
            update_category = 2

            if noise_option == "mallows_noise":
                epsilon = np.random.normal(0, 1)
                mallow_theta_prime = mallow_theta * np.exp(sigma_mallow * epsilon)
                log_prior_current = StatisticalUtils.dTprior(mallow_theta, ua=mallow_ua)
                log_prior_proposed = StatisticalUtils.dTprior(mallow_theta_prime, ua=mallow_ua)
                # Evaluate new likelihood
                log_llk_proposed =HPO_LogLikelihoodCache.calculate_log_likelihood_hpo(
                    U=U,
                    h_U=h_U,
                    observed_orders=observed_orders,
                    M_a_dict=M_a_dict,
                    O_a_i_dict=O_a_i_dict,
                    item_to_index=item_to_index,
                    prob_noise=prob_noise,
                    mallow_theta=mallow_theta_prime,
                    noise_option=noise_option,
                    alpha=alpha
                )

                log_accept_ratio = (
                    (log_prior_proposed + log_llk_proposed)
                    - (log_prior_current + log_llk_current)
                    # Jacobian for multiplicative => log(mallow_theta/mallow_theta_prime)
                    + math.log(mallow_theta / mallow_theta_prime)
                )
                accept_prob = min(1.0, math.exp(min(log_accept_ratio, 700)))

                log_likelihood_primes.append(log_llk_proposed)
                log_likelihood_currents.append(log_llk_current)

                if random.random() < accept_prob:
                    mallow_theta = mallow_theta_prime
                    num_acceptances += 1
                    acceptance_decisions.append(1)
                    log_llk_current = log_llk_proposed
                    accepted_this_iter = True
                else:
                    acceptance_decisions.append(0)

                proposed_mallow_theta_vals.append(mallow_theta_prime)



            elif noise_option == "queue_jump":
                prob_noise_prime = StatisticalUtils.rPprior(noise_beta_prior)
                log_prior_current = StatisticalUtils.dPprior(prob_noise, beta_param=noise_beta_prior)
                log_prior_proposed = StatisticalUtils.dPprior(prob_noise_prime, beta_param=noise_beta_prior)

                # Evaluate new likelihood
                log_llk_proposed = HPO_LogLikelihoodCache.calculate_log_likelihood_hpo(
                    U= U,
                    h_U=h_U,
                    observed_orders=observed_orders,
                    M_a_dict=M_a_dict,
                    O_a_i_dict=O_a_i_dict,
                    item_to_index=item_to_index,
                    prob_noise=prob_noise_prime,
                    mallow_theta=mallow_theta,
                    noise_option=noise_option,
                    alpha=alpha
                )

                log_accept_ratio =log_llk_proposed-log_llk_current
                accept_prob = min(1.0, math.exp(min(log_accept_ratio, 700)))

                log_likelihood_primes.append(log_llk_proposed)
                log_likelihood_currents.append(log_llk_current)

                if random.random() < accept_prob:
                    prob_noise = prob_noise_prime
                    num_acceptances += 1
                    acceptance_decisions.append(1)
                    log_llk_current = log_llk_proposed
                    accepted_this_iter = True
                else:
                    acceptance_decisions.append(0)

                proposed_prob_noise_vals.append(prob_noise_prime)


        elif r <threshold_U0_pct: # Update global latent U0
            update_category = 3
            # Update global latent U0
            n_global = len(M0)
            j_global = np.random.randint(0, n_global-1)
            old_value = U0[j_global, :].copy()

            # We'll do a local random-walk with covariance = Sigma_rho
            Sigma = BasicUtils.build_Sigma_rho(K,rho)
            proposed_row = rng.multivariate_normal(old_value, cov=Sigma)

            # Make a copy
            U0_prime = U0.copy()
            U0_prime[j_global,:] = proposed_row
            U_a_dict_prime={}
            for a, Ua in U_a_dict.items():
                
                Ua_prime = Ua.copy()
                if j_global in M_a_dict[a]:
                    i_loc = M_a_dict[a].index(j_global)
                    old_local = Ua[i_loc, :].copy()
                    local_cov = (1.0 - tau**2) * Sigma_rho
                    np.fill_diagonal(local_cov, 1)
                    new_local_val = rng.multivariate_normal(mean=old_local, cov=local_cov)
                    Ua_prime[i_loc, :] = new_local_val
                U_a_dict_prime[a] = Ua_prime
            U_prime = {"U0": U0_prime, "U_a_dict": U_a_dict_prime}


            # Build partial orders
            h_U_prime = StatisticalUtils.build_hierarchical_partial_orders(
                M0=M0,
                assessors=assessors,
                M_a_dict=M_a_dict,
                U0=U0_prime,
                U_a_dict=U_a_dict_prime,
                alpha=alpha
            )
            # Evaluate new log-likelihood
            log_llk_proposed = HPO_LogLikelihoodCache.calculate_log_likelihood_hpo(
                U=U_prime,
                h_U=h_U_prime,
                observed_orders=observed_orders,
                M_a_dict=M_a_dict,
                O_a_i_dict=O_a_i_dict,
                item_to_index=item_to_index,
                prob_noise=prob_noise,
                mallow_theta=mallow_theta,
                noise_option=noise_option,
                alpha=alpha
            )

            lp_proposed = (
                StatisticalUtils.log_U_prior(U0_prime, rho, K)
                + StatisticalUtils.log_U_a_prior(U_a_dict_prime, tau, rho, K, M_a_dict, U0_prime)
            )
     
            lp_current = (
                StatisticalUtils.log_U_prior(U0, rho, K)
                + StatisticalUtils.log_U_a_prior(U_a_dict, tau, rho, K, M_a_dict, U0)
            )


            # Acceptance ratio
            log_accept_ratio = (lp_proposed + log_llk_proposed) - (lp_current + log_llk_current)
            accept_prob = min(1.0, math.exp(min(log_accept_ratio, 700)))


            if random.random() < accept_prob:
                U0 = U0_prime
                num_acceptances += 1
                acceptance_decisions.append(1)
                log_llk_current = log_llk_proposed
                accepted_this_iter = True
            else:
                acceptance_decisions.append(0)
            proposed_U0.append(U0_prime)



        elif r<threshold_Ua_pct:
            # Assessor-level update.
            update_category = 4 
            a_key = random.choice(assessors)
            M_a = M_a_dict.get(a_key, [])
            if not M_a:
                continue
            n_a = len(M_a)
            row_loc = random.randint(0, n_a - 1)
            old_value = U_a_dict[a_key][row_loc, :].copy()
            Sigma = BasicUtils.build_Sigma_rho(K, rho)
            proposed_row = rng.multivariate_normal(mean=old_value, cov=Sigma)
            U_a_dict_prime = {}
            # Deep copy each assessor's array.
            for a in assessors:
                U_a_dict_prime[a] = U_a_dict[a].copy()
            U_a_dict_prime[a_key][row_loc, :] = proposed_row
            U_prime = {"U0": U0, "U_a_dict": U_a_dict_prime}
            h_U_prime = StatisticalUtils.build_hierarchical_partial_orders(
                M0=M0,
                assessors=assessors,
                M_a_dict=M_a_dict,
                U0=U0,
                U_a_dict=U_a_dict_prime,
                alpha=alpha
            )
            log_llk_proposed = HPO_LogLikelihoodCache.calculate_log_likelihood_hpo(
                U=U_prime,
                h_U=h_U_prime,
                observed_orders=observed_orders,
                M_a_dict=M_a_dict,
                O_a_i_dict=O_a_i_dict,
                item_to_index=item_to_index,
                prob_noise=prob_noise,
                mallow_theta=mallow_theta,
                noise_option=noise_option,
                alpha=alpha
            )
            lp_current = StatisticalUtils.log_U_prior(U0, rho, K) \
                            + StatisticalUtils.log_U_a_prior(U_a_dict, tau, rho, K, M_a_dict, U0)
            lp_proposed = StatisticalUtils.log_U_prior(U0, rho, K) \
                            + StatisticalUtils.log_U_a_prior(U_a_dict_prime, tau, rho, K, M_a_dict, U0)
            log_acceptance_ratio = (lp_proposed + log_llk_proposed) - (lp_current + log_llk_current)
            if random.random() < min(1.0, np.exp(min(log_acceptance_ratio,700))):
                U_a_dict = U_a_dict_prime
                num_acceptances += 1
                acceptance_decisions.append(1)
                log_llk_current = log_llk_proposed
                accepted_this_iter = True
            else:
                acceptance_decisions.append(0)


        # Append current parameter values to trace lists for debugging
        if iteration % 100 == 0:
            rho_trace.append(rho)
            tau_trace.append(tau)
            prob_noise_trace.append(prob_noise)
            mallow_theta_trace.append(mallow_theta)
            U0_trace.append(U0.copy())
            # For U_a_dict, store a deep copy.
            Ua_trace.append(copy.deepcopy(U_a_dict))
            H_trace.append(copy.deepcopy(h_U))
            acceptance_rates.append(num_acceptances / (iteration + 1))
            update_records.append((iteration, update_category, accepted_this_iter))

        current_acceptance_rate = num_acceptances / iteration
        acceptance_rates.append(current_acceptance_rate)

        if iteration in progress_intervals:
            print(f"Iteration {iteration}/{num_iterations} - Accept Rate: {current_acceptance_rate:.2%}")
    overall_acceptance_rate = num_acceptances / num_iterations
    update_df = pd.DataFrame(update_records, columns=["iteration", "category", "accepted"])

    result_dict = {
        "rho_trace": rho_trace,
        "tau_trace": tau_trace,
        "prob_noise_trace": prob_noise_trace,
        "mallow_theta_trace": mallow_theta_trace,
        "U0_trace": U0_trace,
        "Ua_trace": Ua_trace,
        "H_trace": H_trace,
        "proposed_rho_vals": proposed_rho_vals,
        "proposed_tau_vals": proposed_tau_vals,
        "proposed_prob_noise_vals": proposed_prob_noise_vals,
        "proposed_mallow_theta_vals": proposed_mallow_theta_vals,
        "acceptance_decisions": acceptance_decisions,
        "acceptance_rates": acceptance_rates,
        "overall_acceptance_rate": overall_acceptance_rate, 
        "log_likelihood_currents": log_likelihood_currents,
        "log_likelihood_primes": log_likelihood_primes,
        "num_acceptances": num_acceptances,
        # Final state
        "rho_final": rho,
        "tau_final": tau,
        "prob_noise_final": prob_noise,
        "mallow_theta_final": mallow_theta,
        "U0_final": U0,
        "U_a_final": U_a_dict,
        "H_final": h_U,
        "update_df": update_df
    }
    return result_dict
