import numpy as np
from utils import RMSE, MAE, making_null_space, greedy_pair_selection

def construct_rotation_matrix(params, F_rho):
    """
    Construct a rotation matrix.
    """
    rotation_matrix = np.eye(F_rho)
    index = 0
    for i in range(F_rho):
        for j in range(i + 1, F_rho):
            angle = params[index]
            G = np.eye(F_rho)
            G[i, i] = np.cos(angle)
            G[i, j] = -np.sin(angle)
            G[j, i] = np.sin(angle)
            G[j, j] = np.cos(angle)
            rotation_matrix = rotation_matrix @ G
            index += 1
    return rotation_matrix

def objective(params, X_alpha, X_rho, VT_alpha_complement, U_matrix, Sigma_matrix, F_all, F_rho, 
              knowingDataIndex, TrueMaxValues, adversary_config, knowCols):
    """
    Objective function for optimization.
    """
    if adversary_config["SigmaFlag"]:
        rotation_matrix = construct_rotation_matrix(params, F_rho)
        V_rho_candidate = VT_alpha_complement @ rotation_matrix
        X_rho_candidate = U_matrix @ Sigma_matrix @ V_rho_candidate
    else:
        # Calculate singular values
        sigma1 = params[0]
        deltas = params[1:F_all]
        sigmas = np.zeros(F_all)
        sigmas[0] = sigma1
        for i in range(1, F_all):
            sigmas[i] = sigmas[i - 1] - deltas[i - 1]

        # Ensure singular values are positive
        if np.any(sigmas <= 0):
            return 1e6  # Penalize heavily

        rotation_matrix = construct_rotation_matrix(params[F_all:], F_rho)
        V_alpha_complement = making_null_space(X_alpha, sigmas, U_matrix)
        V_rho_candidate = V_alpha_complement @ rotation_matrix
        X_rho_candidate = U_matrix @ np.diag(sigmas) @ V_rho_candidate

    overall_loss = 0
    Loss = eval(adversary_config["Loss"])
    
    # Calculate various losses
    if adversary_config["VarValueFlag"]:
        Variances = np.var(X_rho_candidate, axis=0)
        var_loss = Loss(Variances, np.ones(F_rho))
        overall_loss += var_loss
    
    if adversary_config["MaxValueFlag"]:
        MaxValue = np.max(abs(X_rho_candidate), axis=0)
        reordered_MaxValue = greedy_pair_selection(np.array(TrueMaxValues[:F_rho]).reshape((1, F_rho)), 
                                                   MaxValue.reshape((1, F_rho)), F_rho)
        max_loss = Loss(reordered_MaxValue, TrueMaxValues[:F_rho])
        overall_loss += max_loss

    if adversary_config["MeanValueFlag"]:
        Means = np.mean(X_rho_candidate, axis=0)
        mean_loss = Loss(Means, np.zeros(F_rho))
        overall_loss += mean_loss

    if adversary_config["EntropyFlag"]:
        entropies = []
        for i in range(F_rho):
            variance = np.var(X_rho_candidate[:, i], ddof=1)
            if variance == 0:
                entropies.append(0)
            else:
                entropy = 0.5 + 0.5 * np.log(2 * np.pi * variance)
                entropies.append(entropy)
        overall_loss += (sum(entropies) / F_rho)

    if adversary_config["PointValue"]["Flag"]:
        # columnIndex = np.arange(F_rho)
        trueData = X_rho[knowingDataIndex]
        generatedData = X_rho_candidate[knowingDataIndex]
        reordered_generatedData = greedy_pair_selection(trueData, generatedData, F_rho)
        data_loss = Loss(trueData[:knowCols], reordered_generatedData[:knowCols])
        overall_loss += data_loss * adversary_config['PointValue']['lambda']

    return overall_loss