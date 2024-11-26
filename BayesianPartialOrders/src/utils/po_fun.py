import numpy as np
import math
from scipy.stats import multivariate_normal
import networkx as nx
import random
import itertools
import matplotlib.pyplot as plt

class PO_util:
    @staticmethod
    def seq2dag(seq, n):
        """
        Converts a sequence to a directed acyclic graph (DAG) represented as an adjacency matrix.

        Parameters:
        - seq: A sequence (list) of integers representing a total order.
        - n: Total number of elements.

        Returns:
        - adj_matrix: An n x n numpy array representing the adjacency matrix of the DAG.
        """
        adj_matrix = np.zeros((n, n), dtype=int)
        for i in range(len(seq)):
            u = seq[i] - 1  # Convert to 0-based index
            for j in range(i + 1, len(seq)):
                v = seq[j] - 1  # Convert to 0-based index
                adj_matrix[u, v] = 1
        return adj_matrix

    @staticmethod
    def order2partial(v, n=None):
        """
        Computes the intersection of the transitive closures of a list of total orders.

        Parameters:
        - v: List of sequences, where each sequence is a list of integers representing a total order.
        - n: Total number of elements (optional).

        Returns:
        - result_matrix: An n x n numpy array representing the adjacency matrix of the partial order.
        """
        if n is None:
            n = max(max(seq) for seq in v)
        z = np.zeros((n, n), dtype=int)
        for seq in v:
            dag_matrix = PO_util.seq2dag(seq, n)
            closure_matrix = PO_util.transitive_closure(dag_matrix)
            z += closure_matrix
        result_matrix = (z == len(v)).astype(int)
        return result_matrix

    @staticmethod
    def generate_latent_positions(n, K, rho):
        """
        Generates latent positions Z for n items in K dimensions with correlation rho.

        Parameters:
        - n: Number of items.
        - K: Number of dimensions.
        - rho: Correlation coefficient between dimensions.

        Returns:
        - Z: An n x K numpy array of latent positions.
        """
        Sigma = np.full((K, K), rho)
        np.fill_diagonal(Sigma, 1)
        mu = np.zeros(K)
        rv = multivariate_normal(mean=mu, cov=Sigma)
        Z = rv.rvs(size=n)
        if K == 1:
            Z = Z.reshape(n, 1)
        return Z

    @staticmethod
    def generate_partial_order(Z):
        """
        Generates a partial order h from latent variables Z.

        Parameters:
        - Z: An n x K numpy array of latent positions.

        Returns:
        - h: An n x n numpy array representing the adjacency matrix of the partial order,
             where h[i, j] = 1 if Z[i, :] > Z[j, :] element-wise for all dimensions k.
        """
        n, K = Z.shape
        h = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                if i != j:
                    if np.all(Z[i, :] > Z[j, :]):
                        h[i, j] = 1
        return h

    @staticmethod
    def transitive_reduction(adj_matrix):
        """
        Computes the transitive reduction of a partial order represented by an adjacency matrix.

        Parameters:
        - adj_matrix: An n x n numpy array representing the adjacency matrix of the partial order.

        Returns:
        - tr: An n x n numpy array representing the adjacency matrix of the transitive reduction.
        """
        n = adj_matrix.shape[0]
        tr = adj_matrix.copy()
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if tr[i, k] and tr[k, j]:
                        tr[i, j] = 0
        return tr

    @staticmethod
    def transitive_closure(adj_matrix):
        """
        Computes the transitive closure of a relation represented by an adjacency matrix.

        Parameters:
        - adj_matrix: An n x n numpy array representing the adjacency matrix of the relation.

        Returns:
        - closure: An n x n numpy array representing the adjacency matrix of the transitive closure.
        """
        n = adj_matrix.shape[0]
        closure = adj_matrix.copy()
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    closure[i, j] = closure[i, j] or (closure[i, k] and closure[k, j])
        return closure

    @staticmethod
    def nle(tr):
        """
        Counts the number of linear extensions of the partial order with transitive reduction `tr`.

        Parameters:
        - tr: An n x n numpy array representing the adjacency matrix of the transitive reduction.

        Returns:
        - count: An integer representing the number of linear extensions.
        """
        if tr.size == 0 or len(tr) == 1:
            return 1

        n = tr.shape[0]
        cs = np.sum(tr, axis=0)
        csi = (cs == 0)
        bs = np.sum(tr, axis=1)
        bsi = (bs == 0)
        free = np.where(bsi & csi)[0]
        k = len(free)

        if k == n:
            return math.factorial(n)

        if k > 0:
            tr = np.delete(np.delete(tr, free, axis=0), free, axis=1)
            fac = math.factorial(n) // math.factorial(n - k)
        else:
            fac = 1

        if (n - k) == 2:
            return fac

        tops = np.where(csi)[0]
        bots = np.where(bsi)[0]

        if len(tops) == 1 and len(bots) == 1:
            i = tops[0]
            j = bots[0]
            trr = np.delete(np.delete(tr, [i, j], axis=0), [i, j], axis=1)
            return fac * PO_util.nle(trr)

        count = 0
        for i in tops:
            trr = np.delete(np.delete(tr, i, axis=0), i, axis=1)
            count += PO_util.nle(trr)

        return fac * count

    @staticmethod
    def generate_random_PO(n):
        """
        Generates a random partial order (directed acyclic graph) with `n` nodes.
        Ensures there are no cycles in the generated graph.

        Parameters:
        - n: Number of nodes in the partial order.

        Returns:
        - h: A NetworkX DiGraph representing the partial order.
        """
        h = nx.DiGraph()
        h.add_nodes_from(range(n))
        possible_edges = list(itertools.combinations(range(n), 2))
        random.shuffle(possible_edges)
        for u, v in possible_edges:
            if random.choice([True, False]):
                h.add_edge(u, v)
                if not nx.is_directed_acyclic_graph(h):
                    h.remove_edge(u, v)
        return h

    @staticmethod
    def visualize_partial_order(final_h):
        """
        Visualizes the partial order using NetworkX and Matplotlib.

        Parameters:
        - final_h: An n x n numpy array representing the adjacency matrix of the partial order.
        """
        G = nx.DiGraph()
        n = final_h.shape[0]
        for idx in range(n):
            G.add_node(idx)
        for i in range(n):
            for j in range(n):
                if final_h[i, j] == 1:
                    G.add_edge(i, j)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', arrowsize=20)
        plt.title('Partial Order Graph')
        plt.show()

    @staticmethod
    def log_prior(Z, rho, K):
        """
        Computes the log-prior probability of Z assuming a multivariate normal distribution using SciPy.

        Parameters:
        - Z: An n x K numpy array of latent positions.
        - rho: Correlation coefficient between dimensions.
        - K: Number of dimensions.

        Returns:
        - sum_log_prob: The sum of log-probabilities for all items.
        """
        Sigma = np.full((K, K), rho)
        np.fill_diagonal(Sigma, 1)

        # Create a multivariate normal distribution with mean zero and covariance Sigma
        mvn = multivariate_normal(mean=np.zeros(K), cov=Sigma)
        
        # Compute the log PDF at each row of Z
        log_prob = mvn.logpdf(Z)
        sum_log_prob = np.sum(log_prob)
        
        return sum_log_prob

    @staticmethod
    def log_likelihood(h_Z, n_obs):
        """
        Computes the log-likelihood of the observed data given the partial order h_Z.

        Parameters:
        - h_Z: An n x n numpy array representing the adjacency matrix of the partial order.
        - n_obs: Number of observations (total orders).

        Returns:
        - log_likelihood: The log-likelihood value.
        """
        nle_h_Z = PO_util.nle(h_Z)
        if nle_h_Z == 0:
            return -np.inf  # Log-likelihood is negative infinity if there are zero linear extensions
        else:
            return -n_obs * np.log(nle_h_Z)
