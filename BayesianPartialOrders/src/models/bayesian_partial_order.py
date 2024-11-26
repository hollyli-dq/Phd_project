import numpy as np
from scipy.stats import norm, gumbel_r
from itertools import permutations
import math

class BayesianPartialOrderInference:
    def __init__(self, M, K, rho):
        """
        Initialize the Bayesian model with given parameters.
        :param M: Number of objects
        :param K: Number of features (latent dimensions)
        :param rho: Correlation parameter
        """
        self.M = M  # Number of objects
        self.K = K  # Number of latent features
        self.rho = rho  # Correlation parameter between features
        self.U = np.random.normal(0, 1, (M, K))  # Preference weights sampled from a normal distribution
        self.Sigma = self.construct_covariance_matrix()

    def construct_covariance_matrix(self):
        """Constructs the covariance matrix for the latent variables."""
        Sigma = np.ones((self.K, self.K)) * self.rho
        np.fill_diagonal(Sigma, 1)
        return Sigma

    def compute_eta(self, U):
        """
        Maps the latent variable matrix U into the Gumbel-distributed space.
        This step transforms the normal variables into something appropriate for
        modeling rankings.
        """
        return gumbel_r.ppf(norm.cdf(U))  # Gumbel distribution transformation

    def likelihood(self, y, h):
        """
        Computes the likelihood for the observed ranking y given the partial order h.
        This involves enumerating all linear extensions of the partial order.
        :param y: Observed ranking (tuple of integers)
        :param h: Partial order (dict or similar structure)
        :return: Likelihood value
        """
        # Computing L[h], the set of all linear extensions of h
        L_h = self.get_linear_extensions(h)
        if y in L_h:
            return 1.0 / len(L_h)
        else:
            return 0.0

    def get_linear_extensions(self, h):
        """
        Returns all linear extensions of a given partial order h.
        A linear extension is a total ordering that respects the partial order.
        :param h: A dictionary representing a partial order (e.g., {1: 2, 3: 4} means 1 < 2, 3 < 4)
        :return: List of linear extensions (total orders consistent with h)
        """
        elements = list(h)
        return list(permutations(elements))

    def prior(self):
        """ 
        Prior for U ~ N(0, Σ_rho), where Σ_rho is the covariance matrix.
        Generates the latent preference weights for the Bayesian model.
        """
        return np.random.multivariate_normal(np.zeros(self.K), self.Sigma, self.M)

    def posterior(self, y, h):
        """
        Computes the posterior distribution based on the prior and likelihood.
        The posterior is proportional to the product of the likelihood and the prior.
        :param y: Observed ranking
        :param h: Partial order (poset)
        :return: Posterior value
        """
        prior = self.prior()
        eta = self.compute_eta(prior)
        likelihood = self.likelihood(y, h)
        return likelihood * eta  # Posterior proportional to likelihood * prior

# Example usage
if __name__ == "__main__":
    M = 5  # Number of objects
    K = 2  # Number of features
    rho = 0.5  # Correlation parameter

    # Instantiate the Bayesian Partial Order Inference
    model = BayesianPartialOrderInference(M, K, rho)

    # Example partial order (poset) and observed data y
    h = {1: 2, 3: 4}  # Example poset: 1 < 2, 3 < 4
    y = (1, 2, 3, 4)  # Example observed ranking

    # Compute the posterior
    posterior = model.posterior(y, h)
    print("Posterior:", posterior)
