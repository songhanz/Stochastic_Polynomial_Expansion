import numpy as np
from pprint import pprint
import os
import pandas as pd

nonlinearities = ["relu", "gelu", "tanh"]
# Define dataset URLs and names
dataset_urls = {
    "conc": "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip",
    "ener": "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip",
    "kin8": "https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.arff", 
    "nava": "https://archive.ics.uci.edu/static/public/316/condition+based+maintenance+of+naval+propulsion+plants.zip", 
    "powe": "https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip",
    "prot": "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
    "wine": "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
    "yach": "https://archive.ics.uci.edu/static/public/243/yacht+hydrodynamics.zip"
}

for nonlinearity in nonlinearities:
    print("********************************************")
    print(nonlinearity)
    print("********************************************")
    for dataset, url in dataset_urls.items():
        print(">>>>>>>>>>>>>>>>>>>>>>")
        print(dataset)
        print(">>>>>>>>>>>>>>>>>>>>>>")
        for n_hidden_layer in range(1,5):
            print("============= layer {}".format(n_hidden_layer))
            save_dir = 'UCI_results/' + dataset + '_' + nonlinearity + '_' + str(n_hidden_layer) + 'layer_adapter/'
            
            # Check if the file exists before attempting to open
            if not os.path.exists(save_dir + 'test_perf.txt'):
                print(f"File not found: {save_dir + 'test_perf.txt'}. Skipping this iteration.")
                continue
            else:
                # Read the data
                data = np.loadtxt(save_dir + 'test_perf.txt')
                
                # Check if data is 1D (single line) or 2D (multiple lines)
                if len(data.shape) == 1:
                    # Single line case
                    rmse, logprob = data[0], data[1]
                    print("RMSE: {:.3f}".format(rmse))
                    print("LogProb: {:.3f}".format(logprob))
                else:
                    # Multiple lines case
                    col0 = data[:, 0]
                    col1 = data[:, 1]
                    # Compute means
                    mean_col0 = np.mean(col0)
                    mean_col1 = np.mean(col1)
                    # Compute sample standard deviations (ddof=1 -> sample, not population)
                    std_col0 = np.std(col0, ddof=0)
                    std_col1 = np.std(col1, ddof=0)
                    # Compute standard errors = std / sqrt(N)
                    N = len(col0)
                    stderr_col0 = std_col0 / np.sqrt(N)
                    stderr_col1 = std_col1 / np.sqrt(N)
                    print("RMSE: {:.3f} $\pm$ {:.3f}".format(mean_col0, stderr_col0))
                    print("LogProb: {:.3f} $\pm$ {:.3f}".format(mean_col1, stderr_col1))