import sys
# Add the path to the src directory to access utility modules and po_hm_mcmc.py
sys.path.append('/home/doli/Desktop/research/coding/BayesianPartialOrders/src')  # Adjust the path if your directory structure is different

sys.path.append('/home/doli/Desktop/research/coding/BayesianPartialOrders/src/utils')  # Example path

from po_fun import BasicUtils, StatisticalUtils,GenerationUtils
from po_fun_plot import PO_plot 
from po_accelerator import LogLikelihoodCache 
from po_hm_mcmc_k import mcmc_partial_order

 # Adjust this path based on your directory structure
# Import necessary libraries
import numpy as np
import networkx as nx
import random
import itertools
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import seaborn as sns 
from scipy.stats import beta

import yaml
with open("/home/doli/Desktop/research/coding/BayesianPartialOrders/mcmc_config.yaml", "r") as f:
    config = yaml.safe_load(f)



n = 4 # We want 8 objects 
N= 6 # We want 25 observations
K_prior=config["prior"]["K_prior"]
rho_prior = config["prior"]["rho_prior"]
noise_option = config["noise"]["noise_option"]
mallow_ua = config["prior"]["mallow_ua"]
items = list(range(n))
rho_true=beta.rvs(1,rho_prior)
print("The generated rho true is:")
print(rho_true)

K=StatisticalUtils.rKprior(3, K_prior) 
print(K)
U = GenerationUtils.generate_U(n, K, rho_true)
print("U matrix (latent positions):")
print(U)
h = BasicUtils.generate_partial_order(U)
h_true=BasicUtils.transitive_reduction(h.copy())
print("\nPartial Order (adjacency matrix):")
print(h_true)




print(f"The targeted partial order h is : {h_true}")

StatisticalUtils.description_partial_order(h_true)



items = list(range(n))
item_to_index = {item: idx for idx, item in enumerate(items)}

subsets = GenerationUtils.generate_subsets(N, n)
print(subsets)
## we generate the queue jump noise total order for each subset

h_tc=BasicUtils.transitive_closure(h)



noise_beta_prior = config["prior"]["noise_beta_prior"] 
prob_noise = StatisticalUtils.rPprior(noise_beta_prior)
prob_noise_true= prob_noise
print(prob_noise_true)


import math 

item_to_index = {item: i for i, item in enumerate(items)}

def count_inversions(order, h, item_to_index):
    """
    Counts how many edges in 'h' are violated by 'order'.
    Here, 'order' contains global item indices, h is a global partial order matrix (transitive closure).
    """
    inv = 0
    for i, item_i in enumerate(order):
        for item_j in order[i+1:]:
            # if there should be a relation item_j -> item_i, then inversion happened
            idx_i = item_to_index[item_i]
            idx_j = item_to_index[item_j]
            if h[idx_j, idx_i] == 1:
                inv += 1
    return inv



for subset in subsets:
    total_violations = 0
    num_iterations = 10000  # use sufficient iterations for good average
    for _ in range(num_iterations):
        order = StatisticalUtils.generate_total_order_for_choice_set_with_queue_jump(
            subset, items, h, prob_noise_true
        )
        v = count_inversions(order, BasicUtils.transitive_closure(h),item_to_index)
        total_violations += v

    avg_violations = total_violations / num_iterations
    max_possible_violations = len(subset)*(len(subset)-1)/2  # maximum possible pairwise inversions
    avg_violation_rate = avg_violations / max_possible_violations

    print(f"Subset={subset}, avg violations: {avg_violations:.4f}, "
          f"avg violation rate: {avg_violation_rate:.4f}")
