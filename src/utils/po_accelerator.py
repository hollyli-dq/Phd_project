import sys
import math
import numpy as np
from scipy.stats import beta
import copy 

# Add the path to the utility modules
sys.path.append('/home/doli/Desktop/research/coding/BayesianPartialOrders/src')  # Adjust the path if your directory structure is different

from po_fun import BasicUtils, StatisticalUtils
from mallow_function import Mallows

class LogLikelihoodCache:

    def calculate_log_likelihood(
        Z, h_Z, observed_orders_idx, choice_sets, item_to_index,
        prob_noise, mallow_theta, noise_option
    ):
        if noise_option not in [ "queue_jump", "mallows_noise"]:
            raise ValueError(f"Invalid noise_option: {noise_option}. "
                             f"Valid options are 'no_noise', 'queue_jump', 'mallows_noise'.")


        log_likelihood = 0.0
        # Example loop over data
        for idx, y_i in enumerate(observed_orders_idx):
            O_i = choice_sets[idx]
            O_i_indices = sorted([item_to_index[item] for item in O_i])
            m = len(y_i)

            if noise_option == "queue_jump":
                # Probability of no-jump vs. jump
                for j, y_j in enumerate(y_i):
                    remaining_indices = y_i[j:]
                    h_Z_remaining = h_Z[np.ix_(remaining_indices, remaining_indices)]
                    tr_remaining = BasicUtils.transitive_reduction(h_Z_remaining)
                    num_linear_extensions_remaining = BasicUtils.nle(tr_remaining)

                    local_idx = remaining_indices.index(y_j)
                    num_first_item_extensions = BasicUtils.num_extensions_with_first(tr_remaining, local_idx)

                    prob_no_jump = (1 - prob_noise) * (num_first_item_extensions / num_linear_extensions_remaining)
                    prob_jump = prob_noise * (1 / (m - j))
                    prob_observed = prob_no_jump + prob_jump
                    log_likelihood += math.log(prob_observed)

            elif noise_option == "mallows_noise":
                # Mallows parameter
                h_Z_Oi = h_Z[np.ix_(O_i_indices, O_i_indices)]
                mallows_prob = Mallows.compute_mallows_likelihood(
                    y=y_i,
                    h=h_Z_Oi,
                    theta=mallow_theta,
                    O_i_indice=O_i_indices
                )
                log_likelihood += math.log(mallows_prob if mallows_prob > 0 else 1e-20)

        return log_likelihood



class HPO_LogLikelihoodCache:
    def calculate_log_likelihood_hpo(
                                      U,           # global latent parameters (or assessor-level, as needed)
                                      h_U,         # dictionary: { assessor : { task_index : partial_order_matrix } }
                                      observed_orders,  # dictionary: { assessor : [ observed_order for each task ] }
                                      M_a_dict,
                                      O_a_i_dict,  # dictionary: { assessor : [ choice_set (list of global items) per task ] }
                                      item_to_index,  # mapping for items (if needed; not used in this version)
                                      prob_noise, 
                                      mallow_theta, 
                                      noise_option, 
                                      alpha):
        
        if noise_option not in ["queue_jump", "mallows_noise"]:
            raise ValueError(f"Invalid noise_option: {noise_option}. "
                             f"Valid options are 'no_noise', 'queue_jump', 'mallows_noise'.")
        if observed_orders is None:
            return 0.0



        log_likelihood = 0.0
        # Example loop over data

        
        for a in O_a_i_dict.keys():         
            tasks_choice_sets = O_a_i_dict[a]             # List of subsets for tasks
            tasks_observed = observed_orders.get(a, [])   # Observed orders for assessor a
            tasks_h = h_U.get(a, {})   
            Ma = M_a_dict.get(a, {})  # List of global items for assessor a
            
            for i_task, choice_set in enumerate(tasks_choice_sets):
                sub_size = len(choice_set)
                h_sub = np.zeros((sub_size, sub_size), dtype=int)
                local_map = {item: idx for idx, item in enumerate(choice_set)}

                for r, item_r in enumerate(choice_set):
                    # local index in Ma: where does item_r appear in Ma?
                    local_r = Ma.index(item_r)
                    for c, item_c in enumerate(choice_set):
                        local_c = Ma.index(item_c)
                        h_sub[r, c] = tasks_h[local_r, local_c]


                    # Assume y_i is a list/array of indices (0,1,...,m-1) representing the observed ranking order
                    y_i = tasks_observed[i_task]
                    y_i_local = [local_map[item] for item in y_i if item in local_map]
                    m = len(y_i)               
                    if noise_option == "queue_jump":
                        # Here we assume a mixture: with probability (1 - prob_noise) the ranking is “faithful”
                        # and with probability prob_noise a jump happens.
                        for j, y_j in enumerate(y_i_local):
                            remaining_indices = y_i_local[j:]
                            h_remaining = h_sub[np.ix_(remaining_indices, remaining_indices)] 
                            tr_remaining = BasicUtils.transitive_reduction(h_remaining) 
                            num_linear_extensions_remaining = BasicUtils.nle(tr_remaining)
                            
                            local_idx = remaining_indices.index(y_j)
                            num_first_item_extensions = BasicUtils.num_extensions_with_first(tr_remaining, local_idx)
                            
                            prob_no_jump = (1 - prob_noise) * (num_first_item_extensions / num_linear_extensions_remaining)
                            prob_jump = prob_noise * (1 / (m - j))
                            prob_observed = prob_no_jump + prob_jump
                            log_likelihood += math.log(prob_observed)

                    elif noise_option == "mallows_noise":
                        # Use Mallows model with partial order adjacency h_sub
                        mallows_prob = Mallows.compute_mallows_likelihood(
                            y=y_i,
                            h=h_sub,
                            theta=mallow_theta,
                            O_i_indice=None  # or a custom index set if needed
                        )
                        # Guard against zero
                        log_likelihood += math.log(mallows_prob if mallows_prob > 0 else 1e-20)

        # Cache the current values for U and h_U and the computed log likelihood.

        return log_likelihood
