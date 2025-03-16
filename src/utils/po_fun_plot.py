from collections import Counter
import numpy as np
from scipy.stats import multivariate_normal
import networkx as nx
import seaborn as sns
import pandas as pd  
import itertools
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.stats import beta, kstest
from scipy.integrate import quad
import matplotlib.pyplot as plt
# import pygraphviz as pgv
from scipy.stats import expon, kstest, probplot
import os 

class PO_plot:
    @staticmethod
    def plot_Z_trace(Z_trace, index_to_item):
        """
        Plots the trace of multidimensional latent variables Z over iterations.
        
        Parameters:
        - Z_trace: List of Z matrices over iterations. Each Z is an (n x K) array.
        - index_to_item: Dictionary mapping item indices to item labels.
        """
        Z_array = np.array(Z_trace)  # Shape should be (iterations, n, K)
        iterations = Z_array.shape[0]  # Number of iterations

        # Check dimensions
        if Z_array.ndim != 3:
            raise ValueError("Z_trace should be a list of Z matrices with shape (n, K).")

        _, n_items, K = Z_array.shape  # Extract number of items and latent dimensions

        # Create subplots for each dimension
        fig, axes = plt.subplots(K, 1, figsize=(12, 4 * K), sharex=True)
        if K == 1:
            axes = [axes]  # Ensure axes is iterable when K=1

        for k in range(K):
            ax = axes[k]
            for idx in range(n_items):
                ax.plot(range(iterations), Z_array[:, idx, k], label=f"{index_to_item[idx]}")
            ax.set_ylabel(f'Latent Variable Z (Dimension {k + 1})')
            ax.legend(loc='best', fontsize='small')
            ax.grid(True)
        
        axes[-1].set_xlabel('Iteration')
        plt.suptitle('Trace Plot of Multidimensional Latent Variables Z', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    @staticmethod
    def plot_acceptance_rates(accepted_iterations, acceptance_rates):
        """
        Plots the acceptance rates over iterations.
        
        Parameters:
        - accepted_iterations: List of iteration numbers where acceptance rates were recorded.
        - acceptance_rates: List of acceptance rates corresponding to the iterations.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(accepted_iterations, acceptance_rates, marker='o', linestyle='-', color='blue')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Acceptance Rate', fontsize=12)
        plt.title('Acceptance Rate Over Time', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_top_partial_orders(top_percentages, top_n=5, item_labels=None):
        """
        Plot the top N partial orders as heatmaps with their corresponding frequencies and percentages.
        
        Parameters:
        - top_percentages: List of tuples containing (partial_order_matrix, count, percentage).
        - top_n: Number of top partial orders to plot.
        - item_labels: List of labels for the items. If None, numerical indices are used.
        """
        # Determine the layout of subplots (e.g., 2x3 for 5 plots)
        n_cols = 3  # Number of columns in the subplot grid
        n_rows = (top_n + n_cols - 1) // n_cols  # Ceiling division for rows
        
        plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        
        for idx, (order, count, percentage) in enumerate(top_percentages[:top_n], 1):
            plt.subplot(n_rows, n_cols, idx)
            sns.heatmap(order, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=.5, linecolor='gray',
                        xticklabels=item_labels, yticklabels=item_labels)
            plt.title(f"Top {idx}: {percentage:.2f}%\nCount: {count}")
            plt.xlabel("Items")
            plt.ylabel("Items")
        
        # Remove any empty subplots
        total_plots = n_rows * n_cols
        if top_n < total_plots:
            for empty_idx in range(top_n + 1, total_plots + 1):
                plt.subplot(n_rows, n_cols, empty_idx)
                plt.axis('off')
        
        plt.tight_layout()
        plt.show()



    @staticmethod
    def plot_log_likelihood(mcmc_results: Dict[str, Any], 
                            title: str = 'Log Likelihood Over MCMC Iterations') -> None:
        """
        Plot the total log likelihood over MCMC iterations.
        
        Parameters:
        - mcmc_results (Dict[str, Any]): The dictionary returned by the MCMC function containing log likelihoods.
        - title (str): Title of the plot.
        
        Returns:
        - None. Displays a matplotlib plot.
        """
        
        # Extract log likelihoods
        log_likelihood_currents = mcmc_results.get('log_likelihood_currents', [])
        
        # Calculate total log likelihood for each iteration by summing over observed orders
        # For current states
        # Create an array for iteration numbers
        iterations = np.arange(1, len( log_likelihood_currents) + 1)
        
        # Set the plot style
        sns.set(style="whitegrid")
        plt.figure(figsize=(14, 7))
        
        # Plot total log likelihood for current states
        sns.lineplot(x=iterations, y= log_likelihood_currents, label='Current State', color='blue')
        
        
        # Add titles and labels
        plt.title(title, fontsize=16)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Total Log Likelihood', fontsize=14)
        
        # Add legend
        plt.legend(title='State')
        
        # Enhance layout
        plt.tight_layout()
        
        # Display the plot
        plt.show()
    @staticmethod
    def plot_acceptance_rate(acceptance_rates: List[float], num_iterations: int) -> None:
        """
        Plot the cumulative acceptance rate over MCMC iterations.
        
        Parameters:
        - acceptance_rates (List[float]): Cumulative acceptance rates up to each iteration.
        - num_iterations (int): Total number of iterations.
        
        Returns:
        - None. Displays a matplotlib plot.
        """
        sns.set(style="whitegrid")
        plt.figure(figsize=(14, 7))
        iterations = np.arange(1, num_iterations + 1)
        sns.lineplot(x=iterations, y=acceptance_rates, color='green')
        plt.title('Cumulative Acceptance Rate Over MCMC Iterations', fontsize=16)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Cumulative Acceptance Rate', fontsize=14)
        plt.tight_layout()
        plt.show()
    @staticmethod

    def visualize_partial_order(
        final_h: np.ndarray,
        Ma_list: list,
        title: str = None
    ) -> None:
        """
        Visualizes the partial order for a single assessor using NetworkX and PyGraphviz for layout.
        
        Parameters:
        - final_h (np.ndarray): An n x n numpy array representing the adjacency matrix of the partial order.
        - Ma_list (list): A list of item labels corresponding to the nodes in the partial order.
        - assessor (int, optional): The assessor ID. If provided and title is not specified, it will be used in the default title.
        - title (str, optional): The title of the plot. If not provided, a default title is generated.
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        
        # Set default title if not provided.
        if title is None:
            if assessor is not None:
                title = f"Partial Order Graph for Assessor {assessor}"
            else:
                title = "Partial Order Graph"
        
        # Create a directed graph from the adjacency matrix.
        G = nx.DiGraph(final_h)
        
        # Build node labels from Ma_list.
        # We assume final_h is an n x n matrix with n equal to len(Ma_list).
        labels = {i: str(Ma_list[i]) for i in range(len(Ma_list))}
        
        # Try to use PyGraphviz for layout; if not available, fall back to a spring layout.
        try:
            A = nx.nx_agraph.to_agraph(G)
            # Set node labels using the labels dictionary.
            for node in A.nodes():
                try:
                    node_int = int(node)
                except ValueError:
                    node_int = node
                node.attr['label'] = labels.get(node_int, str(node))
            A.layout('dot')
            A.draw('graph.png')
            img = plt.imread('graph.png')
            plt.imshow(img)
            plt.axis('off')
            plt.title(title)
            plt.show()
        except (ImportError, nx.NetworkXException):
            pos = nx.spring_layout(G)
            nx.draw(G, pos, labels=labels, with_labels=True, arrows=True)
            plt.title(title)
            plt.show()

    @staticmethod
    def visualize_total_orders(total_orders: List[List[int]], top_print: int = 15, top_plot: int = 10) -> None:        
        """
        Visualize the frequency of total orders.

        Parameters:
            total_orders (List[List[int]]): List of total orders, each order is a list of integers.
            top_print (int): Number of top total orders to print.
            top_plot (int): Number of top total orders to plot.

        Returns:
            None. Prints the top_print total orders and displays a bar plot of the top_plot orders.
        """
        
        # 1. Convert total orders to tuples for counting
        total_orders_tuples = [tuple(order) for order in total_orders]
        
        # 2. Count the frequency of each unique total order
        order_counts = Counter(total_orders_tuples)
        
        # 3. Convert tuples to readable strings for better visualization
        total_orders_strings = [' > '.join(map(str, order)) for order in order_counts.keys()]
        frequencies = list(order_counts.values())
        
        # 4. Create a DataFrame from the counter with readable total orders
        df_order_counts = pd.DataFrame({
            'Total Order': total_orders_strings,
            'Frequency': frequencies
        })
        
        # 5. Sort the DataFrame by frequency in descending order
        df_order_counts.sort_values(by='Frequency', ascending=False, inplace=True)
        
        # 6. Reset index for better readability
        df_order_counts.reset_index(drop=True, inplace=True)
        
        # 7. Print the top_print most frequent total orders
        print(f"\nTop {top_print} Most Frequent Total Orders:")
        print(df_order_counts.head(top_print))
        
        # 8. Visualize the frequency counts using Seaborn's barplot
        sns.set(style="whitegrid")  # Set the aesthetic style of the plots
        plt.figure(figsize=(14, 8))  # Set the figure size for better readability
        
        # Create the barplot for top_plot total orders
        sns.barplot(
            x='Total Order',
            y='Frequency',
            data=df_order_counts.head(top_plot),
            palette='viridis'  # Choose a color palette
        )
        
        # Add titles and labels with increased font sizes for clarity
        plt.title(f'Top {top_plot} Most Frequent Total Orders', fontsize=16)
        plt.xlabel('Total Order', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout to prevent clipping of tick-labels
        plt.tight_layout()
        
        # Display the plot
        plt.show()

        @staticmethod   
        def plot_heatmap_and_graph(h_matrix: np.ndarray, title: str, item_labels: Optional[List[str]] = None) -> plt.Figure:
            """
            Create a figure with two subplots:
            - Left: A heatmap of the partial order (h_matrix)
            - Right: A network graph visualization using a spring layout
            
            Parameters:
            h_matrix (np.ndarray): The partial order adjacency matrix.
            title (str): A title for the plots.
            item_labels (list, optional): Labels for items used as tick labels.
            
            Returns:
            fig (plt.Figure): The figure containing the two subplots.
            """
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Left subplot: Heatmap of h_matrix
            sns.heatmap(h_matrix, annot=True, cmap="viridis", 
                        xticklabels=item_labels, yticklabels=item_labels, ax=axes[0])
            axes[0].set_title("Heatmap: " + title)
            axes[0].set_xlabel("Items")
            axes[0].set_ylabel("Items")
            
            # Right subplot: Network graph using spring layout
            G = nx.DiGraph()
            n = h_matrix.shape[0]
            # Add nodes, using item_labels if available
            for idx in range(n):
                label = item_labels[idx] if item_labels and idx < len(item_labels) else str(idx)
                G.add_node(idx, label=label)
            # Add edges from the adjacency matrix
            for i in range(n):
                for j in range(n):
                    if h_matrix[i, j] == 1:
                        G.add_edge(i, j)
                        
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'),
                    node_size=2000, node_color='lightblue', arrowsize=20, ax=axes[1])
            axes[1].set_title("Graph: " + title)
            axes[1].axis('off')
            
            plt.tight_layout()
            return fig
        @staticmethod   
        def plot_mcmc_results(result_dict: Dict[str, Any], pdf_filename: str, item_labels: Optional[List[str]] = None) -> None:
            pp = PdfPages(pdf_filename)
            
            # Use the length of rho_trace for the iteration axis.
            iterations = range(len(result_dict["rho_trace"]))
            
            # --- Page 1: rho trace ---
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(iterations, result_dict["rho_trace"], label="rho", marker='o')
            ax.set_xlabel("Iteration")
            ax.set_ylabel("rho")
            ax.set_title("Trace of rho")
            ax.legend()
            ax.grid(True)
            pp.savefig(fig)
            plt.close(fig)
            
            # --- Page 2: tau trace ---
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(iterations, result_dict["tau_trace"], label="tau", color="orange", marker='o')
            ax.set_xlabel("Iteration")
            ax.set_ylabel("tau")
            ax.set_title("Trace of tau")
            ax.legend()
            ax.grid(True)
            pp.savefig(fig)
            plt.close(fig)
            
            # --- Page 3: Noise parameters ---
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(iterations, result_dict["prob_noise_trace"], label="prob_noise", color="green", marker='o')
            ax.plot(iterations, result_dict["mallow_theta_trace"], label="mallow_theta", color="red", marker='o')
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Noise Parameters")
            ax.set_title("Trace of Noise Parameters")
            ax.legend()
            ax.grid(True)
            pp.savefig(fig)
            plt.close(fig)
            
            # --- Page 4: Log Likelihoods ---
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(iterations, result_dict["log_likelihood_currents"], label="Current Log Likelihood", color="blue", marker='o')
            ax.plot(iterations, result_dict["log_likelihood_primes"], label="Proposed Log Likelihood", 
                    color="purple", marker='o', linestyle="--")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Log Likelihood")
            ax.set_title("Log Likelihood Trace")
            ax.legend()
            ax.grid(True)
            pp.savefig(fig)
            plt.close(fig)
            
            # --- Page 5: Acceptance Rates ---
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(iterations, result_dict["acceptance_rates"], label="Cumulative Acceptance Rate", color="magenta", marker='o')
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Acceptance Rate")
            ax.set_title("Cumulative Acceptance Rate")
            ax.legend()
            ax.grid(True)
            pp.savefig(fig)
            plt.close(fig)
            
            # --- Page 6: Final Partial Orders with Graphs ---
            if "H_final" in result_dict:
                H_final = result_dict["H_final"]
                # Plot global partial order if available (assumed under key 0)
                if 0 in H_final:
                    fig = PO_plot.plot_heatmap_and_graph(H_final[0], "Final Global Partial Order (H0)", item_labels=item_labels)
                    pp.savefig(fig)
                    plt.close(fig)
                # Plot assessor-level partial orders
                for a in H_final:
                    if a == 0:
                        continue
                    value = H_final[a]
                    if isinstance(value, dict):
                        for task, hm in value.items():
                            fig = PO_plot.plot_heatmap_and_graph(hm, f"Assessor {a} - Task {task} Partial Order", item_labels=item_labels)
                            pp.savefig(fig)
                            plt.close(fig)
                    elif isinstance(value, np.ndarray):
                        fig = PO_plot.plot_heatmap_and_graph(value, f"Assessor {a} Partial Order", item_labels=item_labels)
                        pp.savefig(fig)
                        plt.close(fig)
            
            pp.close()
            print(f"Plots saved to {pdf_filename}")


    @staticmethod
    def plot_mcmc_inferred_variables(
        mcmc_results: dict,
        true_param: dict,
        config: dict,
        output_filename: str = "mcmc_inferred_result.pdf",
        output_filepath: str = ".",
        assessors: list = None,
        M_a_dict: dict = None
    ) -> None:
        """
        Creates a series of plots showing:
        (A) Trace plots & histograms for rho, prob_noise, mallow_theta, tau, acceptance_rate, K
        (B) If present, trace/hist for U0[0,0] and local Ua[a][0,0].
        """

        sns.set_style("whitegrid")

        # Extract main MCMC traces
        rho_trace = np.array(mcmc_results["rho_trace"])
        prob_noise_trace = np.array(mcmc_results["prob_noise_trace"])
        mallow_theta_trace = np.array(mcmc_results["mallow_theta_trace"])
        acceptance_rates = np.array(mcmc_results["acceptance_rates"])

        # K trace (if present)
        K_trace = None
        if "K_trace" in mcmc_results and mcmc_results["K_trace"] is not None:
            K_trace = np.array(mcmc_results["K_trace"], dtype=int)

        # Tau trace (if present)
        tau_trace = None
        if "tau_trace" in mcmc_results and mcmc_results["tau_trace"] is not None:
            tau_trace = np.array(mcmc_results["tau_trace"])

        # We'll create 6 rows x 2 columns (the last row for K).
        fig, axes = plt.subplots(6, 2, figsize=(12, 20))
        axes = np.array(axes)
        tol = 1e-4

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # 1) RHO
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        rho_true = true_param.get("rho_true", None)
        ax_rho_trace, ax_rho_hist = axes[0, 0], axes[0, 1]

        # (A) RHO trace
        ax_rho_trace.plot(rho_trace, color="#1f77b4", lw=1.2, alpha=0.8)
        ax_rho_trace.set_title("Trace: rho", fontsize=10)
        ax_rho_trace.set_ylim(0, 1 - tol)
        ax_rho_trace.grid(True, alpha=0.3)

        # (B) RHO histogram
        trunc_point = 1 - tol
        bin_edges = np.linspace(0, trunc_point, 101)
        bin_edges[-1] += 1e-6  # small shift to avoid floating rounding
        sns.histplot(rho_trace, kde=False, ax=ax_rho_hist, color="blue",
                     bins=bin_edges, edgecolor='black', linewidth=0)
        ax_rho_hist.set_title("Histogram: rho", fontsize=10)
        ax_rho_hist.set_xlim(0.5, trunc_point)

        # Overlay truncated Beta(1, config["prior"]["rho_prior"]) if present
        a_ = 1.0
        b_ = config["prior"].get("rho_prior", 1.0)
        x_vals = np.linspace(0.5, trunc_point, 1000)
        norm_const = beta.cdf(trunc_point, a_, b_)
        norm_const = max(norm_const, 1e-15)  # avoid /0
        prior_pdf = beta.pdf(x_vals, a_, b_) / norm_const

        # Plot as black line + label
        ax_rho_hist.plot(x_vals, prior_pdf, 'k-', lw=2, label='Theoretical PDF')

        # Vertical lines
        if rho_true is not None:
            ax_rho_hist.axvline(rho_true, color="red", linestyle="--",
                                label=f"True: {rho_true:.2f}")
        sample_mean_rho = np.mean(rho_trace)
        ax_rho_hist.axvline(sample_mean_rho, color="green", linestyle="--",
                            label=f"Sample: {sample_mean_rho:.2f}")
        ax_rho_hist.legend(fontsize=8, frameon=True)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # 2) PROB_NOISE
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        prob_noise_true = true_param.get("prob_noise_true", None)
        ax_noise_trace, ax_noise_hist = axes[1, 0], axes[1, 1]

        # (A) prob_noise trace
        ax_noise_trace.plot(prob_noise_trace, color="orange")
        ax_noise_trace.set_title("Trace: prob_noise")
        ax_noise_trace.set_xlabel("Iteration")
        ax_noise_trace.set_ylabel("prob_noise")

        # (B) prob_noise histogram
        sns.histplot(prob_noise_trace, kde=False, ax=ax_noise_hist, color="orange")
        ax_noise_hist.set_title("Histogram: prob_noise")
        ax_noise_hist.set_xlabel("prob_noise")

        # Optionally overlay Beta prior
        if "noise_beta_prior" in config["prior"]:
            a_noise = 1.0
            b_noise = config["prior"]["noise_beta_prior"]
            x_vals = np.linspace(0, 1, 500)
            prior_pdf_noise = beta.pdf(x_vals, a_noise, b_noise)
            scale_factor = len(prob_noise_trace) * (ax_noise_hist.get_xlim()[1] / 30.0)
            ax_noise_hist.plot(x_vals, prior_pdf_noise * scale_factor,
                               'k--', label="Beta Prior PDF")
            ax_noise_hist.legend()

        # Vertical lines
        if prob_noise_true is not None:
            ax_noise_hist.axvline(prob_noise_true, color="red", linestyle="--",
                                  label="True Value")
        sample_mean_p = np.mean(prob_noise_trace)
        ax_noise_hist.axvline(sample_mean_p, color="green", linestyle="--",
                              label="Sample Mean")
        ax_noise_hist.legend()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # 3) MALLOW THETA
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        mallow_theta_true = true_param.get("mallow_theta_true", None)
        ax_mtheta_trace, ax_mtheta_hist = axes[2, 0], axes[2, 1]

        # (A) mallow_theta trace
        ax_mtheta_trace.plot(mallow_theta_trace, color="purple")
        ax_mtheta_trace.set_title("Trace: mallow_theta")
        ax_mtheta_trace.set_xlabel("Iteration")
        ax_mtheta_trace.set_ylabel("mallow_theta")

        # (B) mallow_theta histogram
        sns.histplot(mallow_theta_trace, kde=False, ax=ax_mtheta_hist, color="purple")
        ax_mtheta_hist.set_title("Histogram: mallow_theta")
        ax_mtheta_hist.set_xlabel("mallow_theta")

        # Vertical lines
        if mallow_theta_true is not None:
            ax_mtheta_hist.axvline(mallow_theta_true, color="red", linestyle="--",
                                   label="True Value")
        sample_mean_theta = np.mean(mallow_theta_trace)
        ax_mtheta_hist.axvline(sample_mean_theta, color="green", linestyle="--",
                               label="Sample Mean")
        ax_mtheta_hist.legend()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # 4) TAU
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        tau_true = true_param.get("tau_true", None)
        ax_tau_trace, ax_tau_hist = axes[3, 0], axes[3, 1]

        if tau_trace is not None and len(tau_trace) > 0:
            ax_tau_trace.plot(tau_trace, color="brown")
            ax_tau_trace.set_title("Trace: tau")
            ax_tau_trace.set_xlabel("Iteration")
            ax_tau_trace.set_ylabel("tau")

            sns.histplot(tau_trace, kde=False, ax=ax_tau_hist, color="brown")
            ax_tau_hist.set_title("Histogram: tau")
            ax_tau_hist.set_xlabel("tau")

            # Overlaid uniform prior example
            x_vals = np.linspace(0, max(tau_trace)*1.0, 500)
            scale_factor = len(tau_trace)*(ax_tau_hist.get_xlim()[1]/30.0)
            prior_pdf_tau = np.ones_like(x_vals)  # uniform
            ax_tau_hist.plot(x_vals, prior_pdf_tau * scale_factor,
                             'k--', label="Uniform Prior PDF")
            ax_tau_hist.legend()

            # Vertical lines
            if tau_true is not None:
                ax_tau_hist.axvline(tau_true, color="red", linestyle="--",
                                    label="True Value")
            sample_mean_tau = np.mean(tau_trace)
            ax_tau_hist.axvline(sample_mean_tau, color="green", linestyle="--",
                                label="Sample Mean")
            ax_tau_hist.legend()
        else:
            # No tau
            ax_tau_trace.set_title("No tau trace found")
            ax_tau_hist.set_title("No tau trace found")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # 5) ACCEPTANCE RATES
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        ax_acc_trace, ax_acc_hist = axes[4, 0], axes[4, 1]

        ax_acc_trace.plot(acceptance_rates, color="green")
        ax_acc_trace.set_title("Trace: Acceptance Rate")
        ax_acc_trace.set_xlabel("Iteration")
        ax_acc_trace.set_ylabel("Acceptance Rate")

        sns.histplot(acceptance_rates, kde=False, ax=ax_acc_hist, color="green")
        ax_acc_hist.set_title("Histogram: Acceptance Rate")
        ax_acc_hist.set_xlabel("Acceptance Rate")
        ax_acc_hist.set_xlim(0, 1)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # 6) K trace (discrete dimension) if present
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        K_true = true_param.get("K_true", None)
        ax_k_trace, ax_k_hist = axes[5, 0], axes[5, 1]

        if K_trace is not None and len(K_trace) > 0:
            ax_k_trace.plot(K_trace, color="darkcyan", lw=1.2)
            ax_k_trace.set_title("Trace: K", fontsize=10)
            ax_k_trace.set_xlabel("Iteration")
            ax_k_trace.set_ylabel("Dimension K")

            unique_vals, counts = np.unique(K_trace, return_counts=True)
            ax_k_hist.bar(unique_vals, counts, color="darkcyan", edgecolor="black")
            ax_k_hist.set_title("Discrete Values of K")
            ax_k_hist.set_xlabel("K value")
            ax_k_hist.set_ylabel("Frequency")

            lam = config["prior"].get("K_prior", None)  # e.g., truncated Poisson param
            if lam is not None:
                max_k = max(unique_vals)
                k_range = np.arange(1, max_k+3)
                norm_const = 1.0 - math.exp(-lam)
                pmf_vals = []
                for k_ in k_range:
                    log_p = -lam + k_*math.log(lam) - math.lgamma(k_+1)
                    pmf = math.exp(log_p)/norm_const
                    pmf_vals.append(pmf)

                pmf_vals = np.array(pmf_vals)
                scale = len(K_trace)
                ax_k_hist.plot(k_range, pmf_vals*scale, 'k--', label="Trunc. Poisson Prior")
                ax_k_hist.legend()

            if K_true is not None:
                ax_k_hist.axvline(K_true, color="red", linestyle="--",
                                  label=f"K_true={K_true}")
                ax_k_hist.legend()
        else:
            # no K trace
            ax_k_trace.set_title("No K trace found")
            ax_k_hist.set_title("No K trace found")

        # finalize & save
        plt.tight_layout()
        plt.savefig(os.path.join(output_filepath, output_filename))
        print(f"[INFO] Saved MCMC parameter plots to '{output_filename}'")
        plt.show()

        # (C) Optional: Plot U0[0,0]
        U0_trace = mcmc_results.get("U0_trace", None)
        if U0_trace is not None and len(U0_trace) > 0:
            u0_00_vals = [U0[0, 0] for U0 in U0_trace]
            fig_u0, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4))
            ax_left.plot(u0_00_vals, color='blue', marker='.', linestyle='-')
            ax_left.set_title("Trace: U0[0,0]")
            ax_left.set_xlabel("Iteration")
            ax_left.set_ylabel("Value")

            sns.histplot(u0_00_vals, kde=False, ax=ax_right, color='blue')
            ax_right.set_title("Histogram: U0[0,0]")

            out_f = "U0_00_trace_hist.png"
            plt.tight_layout()
            plt.savefig(os.path.join(output_filepath, out_f), dpi=150)
            plt.show()
            print(f"[INFO] Saved U0[0,0] trace & hist to '{out_f}'")
        else:
            print("[INFO] No U0_trace found; skipping U0[0,0] plot.")

        # (D) Optional: Plot local Ua[a][0,0]
        Ua_trace = mcmc_results.get("Ua_trace", None)
        if Ua_trace is not None and assessors and M_a_dict:
            num_iters = len(Ua_trace)
            n_assessors = len(assessors)
            fig_ua, axes_ua = plt.subplots(1, 2*n_assessors, figsize=(8*n_assessors, 4))
            axes_ua = np.array(axes_ua).reshape((1, 2*n_assessors))

            for idx, a in enumerate(assessors):
                if a not in M_a_dict or len(M_a_dict[a]) == 0:
                    print(f"[INFO] Assessor {a} missing or empty M_a_dict.")
                    continue

                local_series = []
                for t_it in range(num_iters):
                    Ua_it_a = Ua_trace[t_it][a]  # shape=(|M_a|, K)
                    if Ua_it_a.shape[0] > 0:
                        local_series.append(Ua_it_a[0, 0]) 
                    else:
                        local_series.append(np.nan)

                ax_left = axes_ua[0, 2*idx]
                ax_left.plot(local_series, color='blue', marker='.', linestyle='-')
                ax_left.set_title(f"Trace: Ua[{a}][0,0]")
                ax_left.set_xlabel("Iteration")
                ax_left.set_ylabel("Value")

                ax_right = axes_ua[0, 2*idx+1]
                clean_vals = [val for val in local_series if not np.isnan(val)]
                sns.histplot(clean_vals, kde=False, ax=ax_right, color='blue')
                ax_right.set_title(f"Histogram: Ua[{a}][0,0]")

            plt.tight_layout()
            out_f = "Ua_by_assessor_trace_hist.png"
            plt.savefig(os.path.join(output_filepath, out_f), dpi=150)
            plt.show()
            print(f"[INFO] Saved Ua[a][0,0] for each assessor to '{out_f}'")
        else:
            print("[INFO] No Ua_trace or no 'assessors'/'M_a_dict'; skipping local U-plot.")