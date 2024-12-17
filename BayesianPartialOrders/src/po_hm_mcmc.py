import sys
import itertools
import random
import numpy as np
from typing import List, Dict, Any

# Add the path to the utility modules
# Adjust this path based on your directory structure
sys.path.append('../src/utils')  # Ensure this path points to the directory containing your utility modules

from po_fun import BasicUtils, StatisticalUtils

def mcmc_partial_order(observed_orders: List[List[int]],
                       choice_sets: List[List[int]],
                       num_iterations: int,
                       Sigma: float,
                       K: int,
                       rho: float) -> Dict[str, Any]:
    """
    Perform MCMC sampling to infer the latent variables Z and the partial order h.

    Parameters:
    - observed_orders (List[List[int]]): List of observed total orders y1, y2, ..., yN.
    - choice_sets (List[List[int]]): List of corresponding subsets O1, O2, ..., ON.
    - num_iterations (int): Number of MCMC iterations.
    - Sigma (float): Standard deviation for proposing new Z.
    - K (int): Number of dimensions for the latent variables Z.
    - rho (float): Step size for proposing new Z.

    Returns:
    - Dict[str, Any]: Dictionary containing samples, acceptance rates, and other relevant data.
    """

    # Map items to indices
    items = sorted(set(itertools.chain.from_iterable(observed_orders)))
    n = len(items)
    item_to_index = {item: idx for idx, item in enumerate(items)}
    index_to_item = {idx: item for item, idx in item_to_index.items()}

    # Convert observed orders to indices
    observed_orders_idx = []
    for order in observed_orders:
        order_idx = [item_to_index[item] for item in order]
        observed_orders_idx.append(order_idx)

    # Initialize Z (latent variables) with zeros
    Z = np.zeros((n, K))

    # Generate initial partial order h_Z from Z
    h_Z = BasicUtils.generate_partial_order(Z)
    print("Initial partial order h_Z:")
    print(h_Z)

    # Initialize storage for samples and diagnostics
    samples_Z = []  # The sampled Z matrices
    proposed_Zs = []  # The proposed Z matrices
    first_acceptance_time = None  # The first iteration where an acceptance occurred
    acceptance_decisions = []  # List indicating acceptance (1) or rejection (0) per iteration
    acceptance_rates = []  # Cumulative acceptance rate up to each iteration
    log_likelihood_currents = []  # Log likelihoods of current states
    log_likelihood_primes = []  # Log likelihoods of proposed states
    h_trace = []  # Trace of h(Z) over iterations

    num_acceptances = 0  # Counter for the number of acceptances

    # Determine intervals for progress updates (e.g., every 10%)
    progress_intervals = [int(num_iterations * frac) for frac in np.linspace(0.1, 1.0, 10)]

    for iteration in range(1, num_iterations + 1):
        # Propose new Z' by modifying one element
        i = random.randint(0, n - 1)  # Select a random row
        k = random.randint(0, K - 1)  # Select a random column
        Z_prime = np.copy(Z)

        # Propose a new value with Gaussian perturbation
        Z_prime[i, k] += np.random.normal(0, Sigma)
        proposed_Zs.append(Z_prime.copy())  # Store the proposed Z

        # Generate partial order for Z'
        h_Z_prime = BasicUtils.generate_partial_order(Z_prime)

        # Initialize variables for log likelihoods
        consistent = True
        log_likelihood_prime = 0.0
        log_likelihood_current = 0.0
        log_current_likelihood_list = []
        log_prime_likelihood_list = []

        # Check consistency of h_Z_prime with observed orders
        consistent = BasicUtils.is_consistent(h_Z_prime, observed_orders_idx)

        if not consistent:
            # If not consistent, reject the proposal
            acceptance_probability = 0
            acceptance_decisions.append(0)
            log_likelihood_currents.append([float('-inf')] * len(observed_orders))
            log_likelihood_primes.append([float('-inf')] * len(observed_orders))
        else:
            # Compute log prior for current and proposed Z
            log_prior_current = StatisticalUtils.log_prior(Z, rho, K)
            log_prior_prime = StatisticalUtils.log_prior(Z_prime, rho, K)

            # Compute log likelihoods for current and proposed states
            for idx, y_i in enumerate(observed_orders_idx):
                O_i = choice_sets[idx]
                O_i_indices = sorted([item_to_index[item] for item in O_i])

                # Extract sub partial orders for the current and proposed Z
                h_Z_Oi = h_Z[np.ix_(O_i_indices, O_i_indices)]
                h_Z_prime_Oi = h_Z_prime[np.ix_(O_i_indices, O_i_indices)]

                # Compute the number of linear extensions
                tr_current = BasicUtils.transitive_reduction(h_Z_Oi)
                num_linear_extensions_current = BasicUtils.nle(tr_current)
                log_likelihood_current += -np.log(num_linear_extensions_current)

                tr_prime = BasicUtils.transitive_reduction(h_Z_prime_Oi)
                num_linear_extensions_prime = BasicUtils.nle(tr_prime)
                log_likelihood_prime += -np.log(num_linear_extensions_prime)

                # Store individual log likelihood contributions
                log_current_likelihood_list.append(-np.log(num_linear_extensions_current))
                log_prime_likelihood_list.append(-np.log(num_linear_extensions_prime))

            # Compute the log acceptance ratio
            log_acceptance_ratio = (log_prior_prime + log_likelihood_prime) - \
                                   (log_prior_current + log_likelihood_current)

            log_likelihood_currents.append(log_current_likelihood_list)
            log_likelihood_primes.append(log_prime_likelihood_list)

            # Calculate acceptance probability
            acceptance_probability = min(1, np.exp(log_acceptance_ratio))

            # Draw a uniform random number to decide acceptance
            rand_val = random.uniform(0, 1)
            if rand_val < acceptance_probability:
                # Accept the proposal
                Z = Z_prime
                h_Z = h_Z_prime
                acceptance_decisions.append(1)
                num_acceptances += 1
                if first_acceptance_time is None:
                    first_acceptance_time = iteration
            else:
                # Reject the proposal
                acceptance_decisions.append(0)

        # Store the current Z and h_Z
        samples_Z.append(Z.copy())
        h_trace.append(h_Z.copy())

        # Calculate and store the cumulative acceptance rate
        current_acceptance_rate = num_acceptances / iteration
        acceptance_rates.append(current_acceptance_rate)

        # Display progress at specified intervals
        if iteration in progress_intervals:
            print(f"Iteration {iteration}/{num_iterations} - Cumulative Acceptance Rate: {current_acceptance_rate:.2%}")

    # Calculate overall acceptance rate
    overall_acceptance_rate = num_acceptances / num_iterations
    print(f"\nOverall Acceptance Rate after {num_iterations} iterations: {overall_acceptance_rate:.2%}")

    return {
        'samples_Z': samples_Z,
        'proposed_Zs': proposed_Zs,
        'index_to_item': index_to_item,
        'item_to_index': item_to_index,
        'acceptance_rates': acceptance_rates,
        'first_acceptance_time': first_acceptance_time,
        'acceptance_decisions': acceptance_decisions,
        'log_likelihood_currents': log_likelihood_currents,
        'log_likelihood_primes': log_likelihood_primes,
        'h_trace': h_trace
    }