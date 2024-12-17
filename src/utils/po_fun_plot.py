from collections import Counter
import numpy as np
from scipy.stats import multivariate_normal
import networkx as nx
import seaborn as sns
import pandas as pd  
import itertools
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any

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
    def visualize_partial_order(final_h, index_to_item):
        """
        Visualizes the partial order as a directed acyclic graph.
        
        Parameters:
        - final_h: Adjacency matrix of the partial order (NumPy array).
        - index_to_item: Dictionary mapping item indices to item labels.
        """
        G = nx.DiGraph()
        n = final_h.shape[0]

        # Add nodes and edges
        for idx in range(n):
            G.add_node(idx, label=index_to_item[idx])
        for i in range(n):
            for j in range(n):
                if final_h[i, j] == 1:
                    G.add_edge(i, j)

        # Generate labels and draw the graph
        labels = {idx: index_to_item[idx] for idx in G.nodes()}
        pos = nx.spring_layout(G)
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000, node_color='lightblue', arrowsize=20)
        plt.title('Partial Order Graph', fontsize=14)
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
        log_likelihood_primes = mcmc_results.get('log_likelihood_primes', [])
        
        # Calculate total log likelihood for each iteration by summing over observed orders
        # For current states
        total_log_likelihood_currents = [sum(likelihoods) for likelihoods in log_likelihood_currents]
        
        # For proposed states (optional, if you want to plot both)
        total_log_likelihood_primes = [sum(likelihoods) for likelihoods in log_likelihood_primes]
        
        # Create an array for iteration numbers
        iterations = np.arange(1, len(total_log_likelihood_currents) + 1)
        
        # Set the plot style
        sns.set(style="whitegrid")
        plt.figure(figsize=(14, 7))
        
        # Plot total log likelihood for current states
        sns.lineplot(x=iterations, y=total_log_likelihood_currents, label='Current State', color='blue')
        
        # Plot total log likelihood for proposed states (optional)
        sns.lineplot(x=iterations, y=total_log_likelihood_primes, label='Proposed State', color='red', alpha=0.5)
        
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
        title: str = 'Partial Order Graph'
    ) -> None:
        """
        Visualizes the partial order using NetworkX and PyGraphviz for layout.

        Parameters:
        - final_h (np.ndarray): An n x n numpy array representing the adjacency matrix of the partial order.
        - title (str): The title of the plot. Defaults to 'Partial Order Graph'.
        """
        G = nx.DiGraph(final_h)

        # Use PyGraphviz to create a graph from NetworkX graph
        try:
            A = nx.nx_agraph.to_agraph(G)
            A.layout('dot')  # 'dot' algorithm is part of Graphviz, similar to Sugiyama
            # Draw the graph with PyGraphviz layout
            A.draw('graph.png')  # Saves the graph to a file
            img = plt.imread('graph.png')
            plt.imshow(img)
            plt.axis('off')  # Turn off axis
            plt.title(title)
            plt.show()
        except (ImportError, nx.NetworkXException):
            # If PyGraphviz is not installed or another error occurs, use a different layout
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, arrows=True)
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
