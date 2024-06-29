import numpy as np
from scipy.optimize import differential_evolution
import pprint
from config_loader import load_config
from data_loader import load_data
from utils import make_directory_if_not_exist, decide_path, making_null_space
from optimization import objective, construct_rotation_matrix
from scipy.linalg import null_space

def main():
    """
    Main function.
    """
    # Load settings from YAML file
    data_config, adversary_config, DE_config = load_config('../configs/attack_config.yaml')
    F_all = data_config['F_all']
    samples = data_config['samples']
    dataset = data_config['dataset']
    num_GT_known = adversary_config['PointValue']['Num']
    num_trials = data_config['numTrials']

    
    path = decide_path(dataset, samples, F_all, adversary_config)
    print("Dest: " + path)
    check = input("Start the explor? (y/n)")
    if check == "n":
        print("abort.")
        return
    path = make_directory_if_not_exist(path, True)

    # Load data
    X_matrix, U_matrix, Sigma_matrix = load_data(samples, F_all, dataset)
    TrueMaxValues = np.max(X_matrix, axis=0)[::-1]

    for F_rho in range(2, F_all):
        F_alpha = F_all - F_rho
        X_alpha = X_matrix[:, :F_alpha]
        X_rho = X_matrix[:, F_alpha:]

        degree_of_freedom = F_rho * (F_rho - 1) // 2
        if F_rho == 1:
            degree_of_freedom = 1

        bounds = [(-np.pi, np.pi) for _ in range(degree_of_freedom)]

        if adversary_config["SigmaFlag"]:
            VT_alpha = np.linalg.inv(Sigma_matrix) @ U_matrix.T @ X_alpha
            VT_alpha_complement = null_space(VT_alpha.T)
        else:
            VT_alpha_complement = None
            Sigma_matrix = None
            bounds = [(90, 120)] + [(0, 30) for _ in range(F_all - 1)] + bounds

        FuncsValues = []
        print(F_rho)
        output_path = f"{path}/{F_rho}-columns"
        make_directory_if_not_exist(output_path, False)
        knowing_data_index_all = np.random.default_rng().integers(0, samples, size=(num_trials, num_GT_known))
        knowCols = adversary_config["PointValue"]["knowCols"]
        if F_rho < float(knowCols):
            knowCols = F_rho

        for record in range(num_trials):
            knowing_data_index = knowing_data_index_all[record]
            print(f"---------- {output_path}, {record}/{num_trials-1} ----------")

            result = differential_evolution(
                objective,
                bounds,
                args=(X_alpha, X_rho, VT_alpha_complement, U_matrix, Sigma_matrix, F_all, F_rho, 
                    knowing_data_index, TrueMaxValues, adversary_config, knowCols),
                strategy=DE_config["strategy"],
                maxiter=DE_config["maxiter"],
                popsize=DE_config["popsize"],
                disp=DE_config["display"]
            )

            if adversary_config["SigmaFlag"]:
                optimal_rotation = construct_rotation_matrix(result.x, F_rho)
                best_SigmaVT_rho = Sigma_matrix @ VT_alpha_complement @ optimal_rotation
            else:
                # Optimized singular values
                sigma1 = result.x[0]
                deltas = result.x[1:F_all]
                sigmas = np.zeros(F_all)
                sigmas[0] = sigma1
                for i in range(1, F_all):
                    sigmas[i] = sigmas[i - 1] - deltas[i - 1]

                optimal_rotation = construct_rotation_matrix(result.x[F_all:], F_rho)
                best_SigmaVT_rho = np.diag(sigmas) @ making_null_space(X_alpha, sigmas, U_matrix) @ optimal_rotation

            np.savetxt(f"{output_path}/try{record}.txt", best_SigmaVT_rho)

            print("-----")
            print(f"{record}/{num_trials-1} finished.")
            pprint.pprint(result)
            FuncsValues.append(result.fun)
            print("-----")

        np.savetxt(f"{output_path}/FunResults.txt", np.array(FuncsValues))
        if adversary_config['PointValue']['Flag']:
            np.savetxt(f"{output_path}/DataPointIndex.txt", knowing_data_index_all, fmt="%5d")

if __name__ == "__main__":
    main()