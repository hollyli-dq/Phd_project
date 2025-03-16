import numpy as np
import math
from scipy.stats import multivariate_normal, norm
import networkx as nx
import random
import seaborn as sns
import pandas as pd
from collections import Counter, defaultdict
import itertools
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any
import sys as sys
import yaml

# Make sure these paths and imports match your local project structure
sys.path.append('/home/doli/Desktop/research/coding/BayesianPartialOrders/src')  # Adjust the path if your directory structure is different

sys.path.append('/home/doli/Desktop/research/coding/BayesianPartialOrders/src/utils')  # Example path
from po_fun import BasicUtils, StatisticalUtils#
from po_fun_plot import PO_plot
from mallow_function import Mallows
from po_accelerator import LogLikelihoodCache,PO_LogLikelihoodCache
from hpo_po_hm_mcmc import mcmc_simulation_hpo
from typing import Dict, List

yaml_file= "/home/doli/Desktop/research/coding/BayesianPartialOrders/hpo_mcmc_configuration.yaml" 

def test_mcmc_simulation_hpo():
    # Sample data
    M0 = [0,1,2,3,4,5]
    assessors = [1,2,3]
    M_a_dict = {
        1:[0,2,4],
        2:[1,2,3,4,5],
        3:[1,2,3,4,5]
    }
    O_a_i_dict = {
        1: [[0,2,4], [0,2]],
        2: [[1,2,5], [1,3], [1,4,5]],
        3: [[1,3], [1,3,4], [2,4,5]]
    }
    observed_orders = {
        1: [
            [0,4,2],
            [0,2]
        ],
        2: [
            [1,5,2],
            [3,1],
            [1,4,5]
        ],
        3: [
            [3,1],
            [4,3,1],
            [5,4,2]
        ]
    }

    alpha = np.array([0.5, -0.2, 0.3, 0.1, 0.0, 1.2]) 
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    # Extract parameters from configuration.
    mcmc_config = config["mcmc"]
    num_iterations = mcmc_config["num_iterations"]
    K = mcmc_config["K"]
    update_probs = mcmc_config["update_probabilities"]
    rho_pct = update_probs["rho"]
    noise_pct = update_probs["noise"]
    tau_pct = update_probs["tau"]
    U_pct = update_probs["U"]
    
    rho_config = config["rho"]
    dr = rho_config["dr"]
    
    noise_config = config["noise"]
    noise_option = noise_config["noise_option"]
    sigma_mallow = noise_config["sigma_mallow"]
    
    prior_config = config["prior"]
    rho_prior = prior_config["rho_prior"]
    noise_beta_prior = prior_config["noise_beta_prior"]
    mallow_ua = prior_config["mallow_ua"]
    
    U_global_update = config["U_global_update"]
    random_seed = config.get("random_seed", 123)

    # call your function
    result = mcmc_simulation_hpo(
        num_iterations=num_iterations,
        M0=M0,
        assessors=assessors,
        M_a_dict=M_a_dict,
        O_a_i_dict=O_a_i_dict,
        observed_orders=observed_orders,
        alpha=alpha,
        K=K ,            # e.g. dimension=2
        dr=dr,
        sigma_mallow=sigma_mallow,
        noise_option=noise_option,
        rho_pct=rho_pct,
        noise_pct=noise_pct,
        tau_pct=tau_pct,
        U_pct=U_pct,
        rho_prior=rho_prior,
        noise_beta_prior=noise_beta_prior,
        mallow_ua=mallow_ua,
        U_global_update=U_global_update,
        random_seed=123
    )
    item_labels = [f"Item {i}" for i in range(len(assessors)+1)]
    PO_plot.plot_mcmc_results(result, pdf_filename="unit_test_mcmc_results.pdf", item_labels=item_labels)
if __name__ == "__main__":
    test_mcmc_simulation_hpo()