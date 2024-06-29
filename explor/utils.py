import os
import numpy as np
from scipy.linalg import null_space

def RMSE(x, y):
    """
    Calculate the Root Mean Squared Error (RMSE).
    """
    return np.sqrt(np.mean((x - y) ** 2))

def MAE(x,y):
    return np.mean(np.abs(x - y))

def making_null_space(X_matrix, s, U_matrix):
    """
    Generate null space.
    """
    return null_space(X_matrix.T @ U_matrix @ np.linalg.inv(np.diag(s)))

def greedy_pair_selection(trueData, generatedData, F_rho):
    """
    Pair true data with generated data using a greedy algorithm. (faster than permutation)
    """
    remaining_indices = list(range(F_rho))
    selected_indices = []

    for i in range(F_rho):
        min_rmse = float('inf')
        best_index = None
        for j in remaining_indices:
            rmse_ = RMSE(trueData[:, i], generatedData[:, j])
            if rmse_ < min_rmse:
                min_rmse = rmse_
                best_index = j
        selected_indices.append(best_index)
        remaining_indices.remove(best_index)

    return generatedData[:, selected_indices]

def make_directory_if_not_exist(path, flag):
    """
    Create a directory if it does not exist.
    """
    if not flag:
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    else:
        time = 1
        while True:
            new_path = f"{path}_{time}"
            if not os.path.exists(new_path):
                os.makedirs(new_path)
                break
            else:
                time += 1
        return new_path

def decide_path(dataset, samples, F_all, adversary_config):
    """
    Determine the path to save results.
    """
    path = f"../outputs/{dataset}/{samples}x{F_all}/"
    make_directory_if_not_exist(path, False)

    if adversary_config["SigmaFlag"]:
        path += "withSigma/"
        make_directory_if_not_exist(path, False)
    else:
        path += "withoutSigma/"
        make_directory_if_not_exist(path, False)
    
    path += (adversary_config["Loss"] + "+")

    if adversary_config["PointValue"]["Flag"]:
        path += f"{adversary_config['PointValue']['Num']}pointsX{adversary_config['PointValue']['knowCols']}"
    if adversary_config["MaxValueFlag"]:
        path += "max"
    if adversary_config["MeanValueFlag"]:
        path += "mean"
    if adversary_config["VarValueFlag"]:
        path += "var"
    if adversary_config["EntropyFlag"]:
        path += "ent"
    return path