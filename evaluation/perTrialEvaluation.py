import numpy as np
import sys
import os

sys.path.append(os.path.abspath('../explor'))
from utils import RMSE, decide_path
from config_loader import load_config, load_config_forEvaluation
from data_loader import load_data


useAttackConfig, data_config, adversary_config = load_config_forEvaluation("../configs/evaluation_config.yaml")
if useAttackConfig:
    data_config, adversary_config, _ = load_config("../configs/attack_config.yaml")

# Input Parameters
F_all = data_config["F_all"]
samples = data_config["samples"]
numTrials = data_config["numTrials"]
dataset = data_config["dataset"]
F_rho = int(input("input the number of F_rho...:"))
F_alpha = F_all - F_rho

# Load Data
X_matrix, U_matrix, Sigma_matrix = load_data(samples, F_all, dataset)

Xalpha = X_matrix[:, :F_alpha]
Xrho = X_matrix[:, F_alpha:]

# Initialize Arrays
orderArray = [i for i in range(F_rho)]

# Get the specific times value for evaluation
times = int(input("input the specific times for evaluation...:"))

path = decide_path(dataset, samples, F_all, adversary_config)

# Load VT_rho_candidate for the specified times
VT_rho_candidate = np.loadtxt(f"{path}/{F_rho}-columns/try{times}.txt")
Xrho_hat = (U_matrix @ VT_rho_candidate).reshape((samples, F_rho))

# Greedy algorithm to find the best permutation
remaining_indices = list(range(F_rho))
selected_indices = []

for i in range(F_rho):
    min_rmse = float('inf')
    best_index = None

    for j in remaining_indices:
        rmse_positive = RMSE(Xrho_hat[:, i], Xrho[:, j])
        rmse_negative = RMSE(-Xrho_hat[:, i], Xrho[:, j])
        rmse_ = min(rmse_positive, rmse_negative)

        if rmse_ < min_rmse:
            min_rmse = rmse_
            best_index = j

    selected_indices.append(best_index)
    remaining_indices.remove(best_index)

# Compute RMSE and PCC with the selected permutation
RMSE_mean = 0
PCC_mean = 0
Xrho_ = Xrho[:, selected_indices]

for i in range(F_rho):
    rmse_ = min(RMSE(Xrho_hat[:, i], Xrho_[:, i]), RMSE(-Xrho_hat[:, i], Xrho_[:, i]))
    RMSE_mean += (rmse_ / F_rho)

    pcc_ = abs(np.corrcoef(np.stack((Xrho_hat[:, i], Xrho_[:, i]), axis=0))[0][1])
    PCC_mean += (pcc_ / F_rho)

# Print and save results
results = f"""
-----------
RMSE: {RMSE_mean:.3f}
PCC : {PCC_mean:.3f}
-----------
"""

print(results)
np.savetxt("../data_save/temp/output.txt", Xrho_hat, fmt="%.10f")