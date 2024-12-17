import numpy as np
import math
from scipy.stats import multivariate_normal
import networkx as nx
import random
import seaborn as sns
import pandas as pd  
from collections import Counter
import itertools
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any
from collections import defaultdict

class ConversionUtils:
    """
    Utility class for converting sequences and orders to different representations.
    """

    @staticmethod
    def seq2dag(seq: List[int], n: int) -> np.ndarray:
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
    def order2partial(v: List[List[int]], n: Optional[int] = None) -> np.ndarray:
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
            dag_matrix = ConversionUtils.seq2dag(seq, n)
            closure_matrix = BasicUtils.transitive_closure(dag_matrix)
            z += closure_matrix
        result_matrix = (z == len(v)).astype(int)
        return result_matrix


class GenerationUtils:
    """
    Utility class for generating latent positions, partial orders, random partial orders, 
    linear extensions, total orders, and subsets.
    """

    @staticmethod
    def generate_latent_positions(n: int, K: int, rho: float) -> np.ndarray:
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
    def generate_random_PO(n: int) -> nx.DiGraph:
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
    def unifLE(tc: np.ndarray, elements: List[int], le: Optional[List[int]] = None) -> List[int]:
        """
        Sample a linear extension uniformly at random from the given partial order matrix `tc`.

        Parameters:
        - tc: Transitive closure matrix representing the partial order (numpy 2D array).
        - elements: List of elements corresponding to the current `tc` matrix.
        - le: List to build the linear extension (default: None).

        Returns:
        - le: A linear extension (list of elements in the original subset).
        """
        if le is None:
            le = []

        if len(elements) == 0:
            return le

        # Find the set of minimal elements (no incoming edges)
        indegrees = np.sum(tc, axis=0)
        minimal_elements_indices = np.where(indegrees == 0)[0]

        if len(minimal_elements_indices) == 0:
            raise ValueError("No minimal elements found. The partial order might contain cycles.")

        # Randomly select one of the minimal elements
        idx_in_tc = random.choice(minimal_elements_indices)
        element = elements[idx_in_tc]
        le.append(element)

        # Remove the selected element from the matrix and elements list
        tc_new = np.delete(np.delete(tc, idx_in_tc, axis=0), idx_in_tc, axis=1)
        elements_new = [e for i, e in enumerate(elements) if i != idx_in_tc]

        # Recursive call
        return GenerationUtils.unifLE(tc_new, elements_new, le)

    @staticmethod
    def sample_total_order(h: np.ndarray, subset: List[int]) -> List[int]:
        """
        Sample a total order (linear extension) for a restricted partial order.

        Parameters:
        - h: The original partial order adjacency matrix.
        - subset: List of node indices to sample a linear extension for.

        Returns:
        - sampled_order: A list representing the sampled linear extension.
        """
        # Restrict the matrix to the given subset
        restricted_matrix = BasicUtils.restrict_partial_order(h, subset)

        # Initialize elements as the elements in the subset
        elements = subset.copy()
        restricted_matrix_tc = BasicUtils.transitive_closure(restricted_matrix)

        # Sample one linear extension using the `unifLE` function
        sampled_order = GenerationUtils.unifLE(restricted_matrix_tc, elements)

        return sampled_order

    @staticmethod
    def generate_subsets(N: int, n: int) -> List[List[int]]:
        """
        Generate N subsets O1, O2, ..., ON where:
        - N is the number of subsets.
        - n is the size of the universal set {0, 1, ..., n-1}.
        
        Each subset Oi is created by:
        - Determining the subset size ni by uniformly sampling from [2, n].
        - Randomly selecting ni distinct elements from the set {0, 1, ..., n-1}.

        Parameters:
        - N: Number of subsets to generate.
        - n: Size of the universal set.

        Returns:
        - subsets: A list of subsets, each subset is a list of distinct integers.
        """
        subsets = []
        universal_set = list(range(n))  # Universal set from 0 to n-1

        for _ in range(N):
            # Randomly sample the subset size ni from [2, n]
            ni = random.randint(2, n)
            # Randomly select ni distinct elements from the universal set
            subset = random.sample(universal_set, ni)
            subsets.append(subset)

        return subsets


class BasicUtils:
    """
    Utility class for basic operations on partial orders.
    """
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
    def restrict_partial_order(h: np.ndarray, subset: List[int]) -> np.ndarray:
        """
        Restrict the partial order matrix `h` to the given `subset`.

        Parameters:
        - h: The original partial order adjacency matrix.
        - subset: List of node indices to restrict to.

        Returns:
        - restricted_matrix: The adjacency matrix restricted to the subset.
        """
        subset_indices = subset  # Elements are already 0-based indices
        restricted_matrix = h[np.ix_(subset_indices, subset_indices)]
        return restricted_matrix

    @staticmethod
    def transitive_reduction(adj_matrix: np.ndarray) -> np.ndarray:
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
    def transitive_closure(adj_matrix: np.ndarray) -> np.ndarray:
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
    def nle(tr: np.ndarray) -> int:
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
            # Delete free rows and columns
            tr = np.delete(np.delete(tr, free, axis=0), free, axis=1)
            fac = math.factorial(n) // math.factorial(n - k)
        else:
            fac = 1

        # Recompute cs and csi based on the updated tr
        cs = np.sum(tr, axis=0)
        csi = (cs == 0)
        bs = np.sum(tr, axis=1)
        bsi = (bs == 0)
        tops = np.where(csi)[0]
        bots = np.where(bsi)[0]

        # Special case: if n - k == 2, return fac
        if (n - k) == 2:
            return fac

        # Check for a unique top and bottom
        if len(tops) == 1 and len(bots) == 1:
            i = tops[0]
            j = bots[0]
            if i < tr.shape[0] and j < tr.shape[1]:
                trr = np.delete(np.delete(tr, [i, j], axis=0), [i, j], axis=1)
                return fac * BasicUtils.nle(trr)
            else:
                return 0  # Or handle appropriately

        # Iterate over all top elements
        count = 0
        for i in tops:
            if i >= tr.shape[0]:
                continue
            trr = np.delete(np.delete(tr, i, axis=0), i, axis=1)
            count += BasicUtils.nle(trr)

        return fac * count

    @staticmethod
    def is_consistent(h: np.ndarray, observed_orders: List[List[int]]) -> bool:
        """
        Check if all observed orders are consistent with the partial order h.

        Parameters:
        - h: The partial order matrix (NumPy array).
        - observed_orders: List of observed total orders (each is a list of item indices).

        Returns:
        - True if all observed orders are consistent with h, False otherwise.
        """
        # Create a directed graph from the partial order matrix h
        G_PO = nx.DiGraph(h)
        # Compute the transitive closure to capture all implied precedence relations
        tc_PO = BasicUtils.transitive_closure(h)

        # Iterate over each observed order
        for idx, order in enumerate(observed_orders):
            # Create a mapping from item to its position in the observed order
            position = {item: pos for pos, item in enumerate(order)}

            # Check all edges in the transitive closure
            for u, v in zip(*np.where(tc_PO == 1)):
                # Check if both u and v are in the observed order
                if u in position and v in position:
                    # If u comes after v in the observed order, it's a conflict
                    if position[u] > position[v]:
                        return False  # Inconsistency found

        return True




class StatisticalUtils:
    """
    Utility class for statistical computations related to partial orders.
    """

    @staticmethod
    def count_unique_partial_orders(h_trace):
        """
        Count the frequency of each unique partial order in h_trace.
        
        Parameters:
        - h_trace: List of NumPy arrays representing partial orders.
        
        Returns:
        - Dictionary with partial order representations as keys and their counts as values.
        """
        unique_orders = defaultdict(int)
        
        for h_Z in h_trace:
            # Convert the matrix to a tuple of tuples for immutability
            h_tuple = tuple(map(tuple, h_Z))
            unique_orders[h_tuple] += 1
    

        sorted_unique_orders = sorted(unique_orders.items(), key=lambda x: x[1], reverse=True)
        
        # Convert the sorted tuples back to NumPy arrays for readability
        sorted_unique_orders = [(np.array(order), count) for order, count in sorted_unique_orders]
        return sorted_unique_orders
    @staticmethod
    def log_prior(Z: np.ndarray, rho: float, K: int, debug: bool = False) -> float:
        """
        Compute the log prior probability of Z.

        Parameters:
        - Z: Current latent variable matrix (numpy.ndarray).
        - rho: Step size for proposal (used here to scale covariance).
        - K: Number of dimensions.
        - debug: If True, prints the covariance matrix.

        Returns:
        - log_prior: Scalar log prior probability.
        """
        # Covariance matrix is scaled identity matrix
        Sigma = rho * np.identity(K)

        if debug:
            print(f"Covariance matrix Sigma:\n{Sigma}")

        # Compute log prior for each row in Z assuming independent MVN
        try:
            mvn = multivariate_normal(mean=np.zeros(K), cov=Sigma)
            log_prob = mvn.logpdf(Z)
            log_prior = np.sum(log_prob)
        except np.linalg.LinAlgError as e:
            print(f"LinAlgError in log_prior: {e}")
            print(f"Covariance matrix Sigma:\n{Sigma}")
            raise e

        return log_prior

    @staticmethod
    def description_partial_order(h: np.ndarray) -> Dict[str, Any]:
        """
        Provides a detailed description of the partial order represented by the adjacency matrix h.

        Parameters:
        - h: An n x n numpy array representing the adjacency matrix of the partial order.

        Returns:
        - description: A dictionary containing descriptive statistics of the partial order.
        """
        G = nx.DiGraph(h)
        n = h.shape[0]
        node_num= G.number_of_nodes()

        # Number of relationships (edges)
        num_relationships = G.number_of_edges()

        # Number of alone nodes (no incoming or outgoing edges)
        alone_nodes = [node for node in G.nodes() if G.in_degree(node) == 0 and G.out_degree(node) == 0]
        num_alone_nodes = len(alone_nodes)

        # Maximum number of relationships a node can have with other nodes
        # Considering both in-degree and out-degree
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        max_in_degree = max(in_degrees.values()) if in_degrees else 0
        max_out_degree = max(out_degrees.values()) if out_degrees else 0
        max_relationships = max(max_in_degree, max_out_degree)

        # Number of linear extensions
        tr = BasicUtils.transitive_reduction(h)
        num_linear_extensions = BasicUtils.nle(tr)

        # Depth of the partial order (length of the longest chain)
        try:
            depth = nx.dag_longest_path_length(G)
        except nx.NetworkXUnfeasible:
            depth = None  # If the graph is not a DAG

        description = {
            "Number of Nodes": node_num,
            "Number of Relationships": num_relationships,
            "Number of Alone Nodes": num_alone_nodes,
            "Alone Nodes": alone_nodes,
            "Maximum In-Degree": max_in_degree,
            "Maximum Out-Degree": max_out_degree,
            "Maximum Relationships per Node": max_relationships,
            "Number of Linear Extensions": num_linear_extensions,
            "Depth of Partial Order": depth
        }

        # Print the description
        print("\n--- Partial Order Description ---")
        for key, value in description.items():
            print(f"{key}: {value}")
        print("---------------------------------")