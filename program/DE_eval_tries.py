import pprint
import numpy as np
from scipy.optimize import differential_evolution
import itertools

'''choice'''
numFeature = 9
target = int(input("input the number of target...:"))
n = 4000

def RMSELoss(yhat,y):
    return np.sqrt(np.mean((yhat-y)**2))

U = np.loadtxt("data_save/temp/Xs_fed_4000x9.txt")
Sigma = np.loadtxt("data_save/temp/sigma.txt")
VTs = np.loadtxt("data_save/temp/VTs.txt")
W = np.loadtxt("data_save/temp/W.txt")
X = np.loadtxt("data_save/temp/X_shared_4000x9.txt")
Xalpha = X[:,:numFeature-target]
Xrho = X[:,numFeature-target:]

orderArray = [i for i in range(target)]
AllarrPCC = []
AllarrRMSE = []

for times in range(20):
	arrPCC = []
	arrRMSE = []
	W_hat = np.loadtxt(f"data_save/evaluation/4000x9-mimic-DE/{target}-columns/{times}.txt")
	Xrho_hat = np.reshape(U@W_hat,(n,target))
	for perm in itertools.permutations(orderArray):
		PCC_mean = 0
		RMSE_mean = 0
		Xrho_ = Xrho[:,perm]
		for i in range(target):
			rmse_ = min(RMSELoss(Xrho_hat[:,i], Xrho_[:,i]), RMSELoss(-Xrho_hat[:,i], Xrho_[:,i]))
			RMSE_mean += (rmse_ / target)

			pcc_ = abs(np.corrcoef(np.stack((Xrho_hat[:,i], Xrho_[:,i]),axis=0))[0][1])
			PCC_mean += (pcc_ / target)
		
		arrRMSE.append(RMSE_mean)
		arrPCC.append(PCC_mean)

	np_arrRMSE = np.array(arrRMSE)
	bestIndex = np.argmin(np_arrRMSE)

	AllarrRMSE.append(arrRMSE[bestIndex])
	AllarrPCC.append(arrPCC[bestIndex])

np.savetxt(f"data_save/evaluation/4000x9-mimic-DE/{target}-columns/RMSEsummary.txt", AllarrRMSE, fmt="%.18f")
np.savetxt(f"data_save/evaluation/4000x9-mimic-DE/{target}-columns/PCCsummary.txt", AllarrPCC, fmt="%.18f")

print("-----------")
print(f"RMSE: {AllarrRMSE}")
print(f"PCC: {AllarrPCC}")
# print((np.diag(Sigma)@VTs)[:,numFeature-target:])
# print(target2)
