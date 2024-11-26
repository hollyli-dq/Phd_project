import     

class PO_plot:
    def plot_Z_trace(Z_trace, index_to_item):
        """
        Plots the trace of multidimensional latent variables Z over iterations.
        
        Parameters:
        - Z_trace: List of Z matrices over iterations. Each Z is an (n x K) array.
        - index_to_item: Dictionary mapping item indices to item labels.
        """
        n = len(index_to_item)
        iterations = len(Z_trace)
        Z_array = np.array(Z_trace)  # Shape should be (iterations, n, K)
        
        # Check that Z_array has the expected dimensions
        if Z_array.ndim != 3:
            raise ValueError("Z_trace should be a list of Z matrices with shape (n, K).")
        
        _, n_items, K = Z_array.shape  # Extract the number of items and dimensions
        
        # Create subplots for each latent dimension
        fig, axes = plt.subplots(K, 1, figsize=(12, 4 * K), sharex=True)
        if K == 1:
            axes = [axes]  # Ensure axes is iterable when K=1
        
        for k in range(K):
            ax = axes[k]
            for idx in range(n_items):
                ax.plot(range(iterations), Z_array[:, idx, k], label=f"{index_to_item[idx]}")
            ax.set_ylabel(f'Latent Variable Z (Dimension {k + 1})')
            ax.legend()
            ax.grid(True)
        
        axes[-1].set_xlabel('Iteration')
        plt.suptitle('Trace Plot of Multidimensional Latent Variables Z')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_acceptance_rates(accepted_iterations, acceptance_rates):
        """
        Plots the acceptance rates over time.
        """
        plt.plot(accepted_iterations, acceptance_rates, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Acceptance Rate')
        plt.title('Acceptance Rate Over Time')
        plt.show()


    def visualize_partial_order(final_h, index_to_item):
        """
        Visualizes the partial order using networkx and matplotlib.
        """
        G = nx.DiGraph()
        n = final_h.shape[0]
        # Add nodes
        for idx in range(n):
            G.add_node(idx, label=index_to_item[idx])
        # Add edges
        for i in range(n):
            for j in range(n):
                if final_h[i, j] == 1:
                    G.add_edge(i, j)
        # Create labels
        labels = {idx: index_to_item[idx] for idx in G.nodes()}
        # Draw the graph
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000, node_color='lightblue', arrowsize=20)
        plt.title('Partial Order Graph')
        plt.show()

