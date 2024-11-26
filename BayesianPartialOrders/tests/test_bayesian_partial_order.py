import unittest
from src.models.bayesian_partial_order import BayesianPartialOrderInference

class TestBayesianPartialOrderInference(unittest.TestCase):
    def setUp(self):
        self.model = BayesianPartialOrderInference(M=5, K=2, rho=0.5)

    def test_prior_generation(self):
        prior = self.model.prior()
        self.assertEqual(prior.shape, (5, 2))

    def test_likelihood(self):
        h = {1: 2, 3: 4}
        y = (1, 2, 3, 4)
        likelihood = self.model.likelihood(y, h)
        self.assertTrue(0 <= likelihood <= 1)

if __name__ == "__main__":
    unittest.main()
