import pprint
import numpy as np
from scipy.optimize import differential_evolution
import os

def RMSELoss(yhat,y):
    return np.sqrt(np.mean((yhat-y)**2))

def U_loss(yhat, y):
    height,width = yhat.shape
    loss = 0
    for w in range(width):
        loss1 = RMSELoss(yhat[:,w], y[:,w])
        loss2 = RMSELoss(yhat[:,w], -y[:,w])
        loss += min(loss1, loss2)
    return loss

numFeature = 9
# target = 2
n = 4000
times = 3

U = np.loadtxt("data_save/temp/Xs_fed_4000x9.txt")
Sigma = np.loadtxt("data_save/temp/sigma.txt")
VTs = np.loadtxt("data_save/temp/VTs.txt")
X = np.loadtxt("data_save/temp/X_shared_4000x9.txt")
Sigma = np.loadtxt("data_save/temp/sigma.txt")

U_mean = np.mean(U)
Sigma_mean = np.mean(Sigma)
U_coef = Sigma_mean / (U_mean + Sigma_mean)
Sigma_coef = U_mean / (U_mean + Sigma_mean)


for target in range(1,numFeature):
    print(f"---------- target={target} ----------")
    Xalpha = X[:,:numFeature-target]
    Xrho = X[:,numFeature-target:]
    bounds = [(-50,50) for i in range(numFeature * target)]
    def func(x):       
        x_ = np.reshape(x,(numFeature,target))
        Xrho_hat = U@x_
        Xo_hat = np.concatenate((Xalpha,Xrho_hat), axis=1)
        U_hat, Sigma_hat, VT_hat = np.linalg.svd(Xo_hat)
        U_hat = U_hat[:,:numFeature]
        # return U_coef*U_loss(U_hat, U) + Sigma_coef*RMSELoss(Sigma_hat, Sigma) 
        return U_loss(U_hat[:,:6], U[:,:6])

    result = differential_evolution(func, bounds, disp=True, workers=-1, maxiter=150, popsize=15)
    pprint.pprint(result)

    newDir2 = f"data_save/evaluation/4000x9-mimic-DE/{target}-columns"
    if not os.path.exists(newDir2):
        os.makedirs(newDir2)
    np.savetxt(f"data_save/evaluation/4000x9-mimic-DE/{target}-columns/{times}.txt", np.fliplr(np.reshape(result.x, (numFeature, target))), fmt="%.18f")

