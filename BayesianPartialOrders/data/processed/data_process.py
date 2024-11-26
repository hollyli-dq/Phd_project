import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

pandas2ri.activate()

# Load the RData file
robjects.r['load']('/home/doli/Desktop/research/coding/BayesianPartialOrders/data/raw/sample_data.RData')

# Access the loaded data by its name in the R environment (e.g., 'sample_data')
# Replace 'sample_data' with the actual variable name inside your RData file
your_dataset = robjects.r['sample_data']

# Convert to pandas DataFrame
dataset = pandas2ri.ri2py(your_dataset)
print(dataset.head())