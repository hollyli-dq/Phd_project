import numpy as np

def load_data(filename):
    """
    Load data from a file (e.g., CSV) to initialize the Bayesian model.
    :param filename: Path to the file
    :return: Processed data (e.g., list of rankings or observations)
    """
    return np.loadtxt(filename, delimiter=",")
