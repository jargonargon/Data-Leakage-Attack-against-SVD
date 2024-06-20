import numpy as np

def load_data(samples, F_all, dataset):
    """
    Load data from text files.
    """
    X_matrix = np.loadtxt(f"../data_save/temp/X_shared_{samples}x{F_all}.txt")
    U_matrix = np.loadtxt(f"../data_save/temp/Xs_fed_{samples}x{F_all}.txt")
    Sigma_matrix = np.diag(np.loadtxt("../data_save/temp/sigma.txt"))
    return X_matrix, U_matrix, Sigma_matrix