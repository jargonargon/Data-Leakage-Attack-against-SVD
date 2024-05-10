import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from utils import *
from livelossplot import PlotLosses
import os
import atexit
import matplotlib as mpl
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import heapq
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

start = time.time()

def is_under_ssh_connection():
    # The environment variable `SSH_CONNECTION` exists only in the SSH session.
    return 'SSH_CONNECTION' in os.environ.keys()

def use_WebAgg(port = 8000, port_retries = 50):
    """use WebAgg for matplotlib backend.
    """
    current_backend = mpl.get_backend()
    current_webagg_configs = {
        'port': mpl.rcParams['webagg.port'],
        'port_retries': mpl.rcParams['webagg.port_retries'],
        'open_in_browser': mpl.rcParams['webagg.open_in_browser'],
    }
    def reset():
        mpl.use(current_backend)
        mpl.rc('webagg', **current_webagg_configs)
    
    mpl.use('WebAgg')
    mpl.rc('webagg', **{
        'port': port,
        'port_retries': port_retries,
        'open_in_browser': False
    })
    atexit.register(reset)


if is_under_ssh_connection():
    use_WebAgg(port = 8000, port_retries = 50)

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

def MSELoss(yhat,y):
    return torch.mean((yhat - y) ** 2)

def U_loss(yhat, y):
    height,width = yhat.shape
    loss = 0
    for w in range(width):
        loss1 = RMSELoss(yhat[:,w], y[:,w])
        loss2 = RMSELoss(yhat[:,w], -y[:,w])
        loss += min(loss1, loss2)
    return loss


task_config = load_task_config()


def train(target, iter, times, row, col, dataset_name):
    """parameter"""
    p = target #passive party's features (target for inference)
    f = col #features of X^o
    n = row

    """ data load"""
    device = torch.device('cpu')



    """model definition"""
    class generator(nn.Module):
        def __init__(self):
            super(generator, self).__init__()
            self.fc1 = nn.Linear(f, p, bias=False)     
            # nn.init.uniform_(self.fc1.weight, a=-0.1, b=0.1) # 重みの初期値を設定
            # nn.init.uniform_(self.fc2.weight, a=-0.1, b=0.1) # 重みの初期値を設定
            # nn.init.uniform_(self.fc3.weight, a=-0.1, b=0.1) # 重みの初期値を設定
            # nn.init.uniform_(self.fc4.weight, a=-0.1, b=0.1) # 重みの初期値を設定
            # nn.init.uniform_(self.fc5.weight, a=-1, b=1) # 重みの初期値を設定
            # nn.init.uniform_(self.fc6.weight, a=-1, b=1) # 重みの初期値を設定
            # nn.init.zeros_(self.fc1.bias)    # バイアスの初期値を設定
            # nn.init.uniform_(self.fc2.bias, a=-0.1, b=0.1)    # バイアスの初期値を設定
            # nn.init.uniform_(self.fc3.bias, a=-0.1, b=0.1)    # バイアスの初期値を設定
            # nn.init.uniform_(self.fc4.bias, a=-0.1, b=0.1)    # バイアスの初期値を設定
            # nn.init.uniform_(self.fc5.bias, a=-0.1, b=0.1)    # バイアスの初期値を設定
            # nn.init.uniform_(self.fc6.bias, a=-0.1, b=0.1)    # バイアスの初期値を設定
        def forward(self, x):
            x = self.fc1(x)
            return nn.Identity()(x)

    """training"""
    epochs = 1000
    lr = 1
    loss = nn.L1Loss(reduction="mean")
    # loss = U_loss

    G = generator().to(device)
    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    scheduler = ReduceLROnPlateau(G_optimizer, mode='min', min_lr=1e-2, verbose=True, patience=10)



    # XA = torch.from_numpy(XA.astype(np.float32)).clone().to(device)

    # U_coef = U[0:f,:]
    history_loss = []
    history_corr = []
    # XA_part = XA[:f,:].T
    #inputs = XA_part.reshape(1,f*p)
    #print(inputs)
    #inputs = torch.randn(1,p*n)
    max_corr = 0
    min_loss = 100
    #print(list(G.parameters()))
    rng = np.random.default_rng()
    epoch1 = rng.integers(100,300)
    epoch2 = rng.integers(500,800)

    for epoch in range(epochs):
        sortlist = np.array([i for i in range(row)])
        rng1 = np.random.default_rng()
        seeds = rng1.integers(0,1000)
        rng2 = np.random.default_rng(seeds)
        sortlist_shuffled = rng2.permutation(sortlist)

        X = np.loadtxt(f'data_save/temp/X_shared_{row}x{col}.txt', delimiter=' ')[sortlist_shuffled]
        U = np.loadtxt(f'data_save/temp/Xs_fed_{row}x{col}.txt', delimiter=' ')[sortlist_shuffled]
        XB = X[:n,0:f-p] #X = XB + XA
        XA = X[:n,f-p:]
        XB = torch.from_numpy(XB.astype(np.float32)).clone().to(device)
        U = torch.from_numpy(U.astype(np.float32)).clone().to(device)
        inputs = U.to(device)

        # inputs = inputs_original
        # inputs = torch.randn(1,p*n)
        XA_tilde = []
        for i in range(n):
            XA_tilde_ = G(inputs[i])

            XA_tilde.append(XA_tilde_)
        XA_tilde = torch.stack(XA_tilde, dim=0)

        #print(f"XA_tilde: {XA_tilde.shape}")
        #print(f"XB: {XB.shape}")
        # 結合, SVD
        X_tilde = torch.cat((XB, XA_tilde), dim=1)
        X_tilde_forSVD = X_tilde
        #print(X_tilde.shape)
        U_tilde, Sigma_tilde, VT_tilde = torch.linalg.svd(X_tilde_forSVD)
        U_tilde = U_tilde[:, :f]
        X_for_loss = U@torch.diag(Sigma_tilde)@VT_tilde
        X_for_loss_A = X_for_loss[:, f-p:]
        #print(X_for_loss_A.shape)
        #print(XA_tilde.shape)

        # 訓練
        #print(U_tilde.shape)
        #G_loss = loss(XA_tilde, X_for_loss_A) # 生成部分のみで損失を計算 (pred,target)
        U_compa = U[:n,:]
        G_loss = loss(U_tilde, U_compa) # 生成部分のみで損失を計算 (pred,target)
        # if epoch==0:
        #     print(G_loss)
        npXA_tilde = XA_tilde.to('cpu').detach().numpy().copy()
        arrCorr = []
        for i in range(p):
            s1 = pd.Series(npXA_tilde[:,i].reshape(n))
            s2 = pd.Series(XA[:,i].reshape(n))
            corr = abs(s1.corr(s2))
            arrCorr.append(corr)

        if min_loss > G_loss.item():
            min_loss = G_loss.item()
            if save_flag:
                np.savetxt(f'data_save/evaluation/{row}x{col}-{dataset_name}-random-msa/{p}-columns/{iter}-X_tilde.txt',X_tilde.to('cpu').detach().numpy().copy(), fmt=task_config['output_format'])
                np.savetxt(f'data_save/evaluation/{row}x{col}-{dataset_name}-random-msa/{p}-columns/{iter}-X.txt',X, fmt=task_config['output_format'])
                np.savetxt(f'data_save/evaluation/{row}x{col}-{dataset_name}-random-msa/{p}-columns/{iter}-U_tilde.txt',U_tilde.to('cpu').detach().numpy().copy(), fmt=task_config['output_format'])
                np.savetxt(f'data_save/evaluation/{row}x{col}-{dataset_name}-random-msa/{p}-columns/{iter}-U.txt',U.to('cpu').detach().numpy().copy(), fmt=task_config['output_format'])
                np.savetxt(f'data_save/evaluation/{row}x{col}-{dataset_name}-random-msa/{p}-columns/{iter}-sigma_tilde.txt',Sigma_tilde.to('cpu').detach().numpy().copy(), fmt=task_config['output_format'])
                np.savetxt(f'data_save/evaluation/{row}x{col}-{dataset_name}-random-msa/{p}-columns/{iter}-VT_tilde.txt',VT_tilde.to('cpu').detach().numpy().copy(), fmt=task_config['output_format'])
                torch.save(G.state_dict(), f'data_save/evaluation/{row}x{col}-{dataset_name}-random-msa/{p}-columns/{iter}-model.pth')
            # print(f" / best corr: {arrCorr} / saved. ") 
            best_correlation = arrCorr
        # history_corr.append(np.mean(arrCorr))
        # history_loss.append(G_loss.item())
        # if (epoch+1)%50 == 0:
        #     print(f" / current corr: {arrCorr}")
        
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()
        # scheduler.step(G_loss)
        if epoch == epoch1 or epoch == epoch2:
            G_optimizer.param_groups[0]['lr'] = G_optimizer.param_groups[0]['lr'] / 10

        end = time.time()
        print("\r"+"epochs: "+str(epoch+1)+"/ " + "all time: " + str(int((end-start)/3600)) + "h  " + str(int(((end-start)%3600)/60)) + "min " + str(int((end-start)%60)) + "sec" + "/ " + str(G_loss),end="")
    return best_correlation, min_loss

save_flag = False
choice = input("wanna save the results? (y/n)")
if choice == "y":
    print("will be saved")
    save_flag = True

#### choice
times = 2
XFeature = [9]
# XFeature = [3,7,11]
Xrows = [4000]
numTry = 100
numCandidate = [20,5]
dataset_name = "mimic"


for XFea in XFeature:
    for row in Xrows:
        newDir1 = f"data_save/evaluation/{row}x{XFea}-{dataset_name}--msa"
        if not os.path.exists(newDir1):
            os.makedirs(newDir1)
        for target in range(1,XFea):
            newDir2 = f"data_save/evaluation/{row}x{XFea}-{dataset_name}-random-msa/{target}-columns"
            if not os.path.exists(newDir2):
                os.makedirs(newDir2)
            allCorr = []
            allLoss = []
            print(f"-------- target={target}, samples={row}, Feature = {XFea} --------")
            for iter in range(numTry):
                corr_, loss_ = train(target,iter,times,row,XFea,dataset_name)
                allCorr.append(corr_)
                allLoss.append(loss_)
                print("")
                print(f" >>> {iter}-th correlation: {corr_}")
                print(f" >>> {iter}-th loss: {loss_}")
            allCorr_ = np.array(allCorr)
            allLoss_ = np.array(allLoss)
            print(f"Best of all: {allCorr[np.argmin(allLoss_)]} at {np.argmin(allLoss_)}-th iteration.")
            if save_flag:
                np.savetxt(f'data_save/evaluation/{row}x{XFea}-{dataset_name}-random-msa/{target}-columns/CorrSummary.txt',allCorr_, fmt=task_config['output_format'])
                np.savetxt(f'data_save/evaluation/{row}x{XFea}-{dataset_name}-random-msa/{target}-columns/LossSummary.txt',allLoss_, fmt=task_config['output_format'])

    # # log output
    # sys.stdout = open(f"data_save/view/{dataset_name}-Nx{XFea}.log", "w")
    # for num in numCandidate:
    #     print("")
    #     print(f"========== {dataset_name} / numCandidate: {num} ==========")
    #     for row in Xrows:
    #         print("")
    #         print(f"------{row}x{XFea}------")
    #         for target in range(1,XFea):
    #             CorrSummary = np.loadtxt(f'data_save/evaluation/{row}x{XFea}-{dataset_name}/{target}-columns/CorrSummary.txt').reshape(numTry,target)
    #             LossSummary = np.loadtxt(f'data_save/evaluation/{row}x{XFea}-{dataset_name}/{target}-columns/LossSummary.txt').reshape(numTry).tolist()
    #             minLossValue = heapq.nsmallest(num, LossSummary)
    #             lstCorrMatrix = []
    #             for value in minLossValue:
    #                 lstCorrMatrix.append(CorrSummary[LossSummary.index(value)])
    #             CorrMatrix = np.array(lstCorrMatrix)
    #             print(f">>> target: {target}")
    #             print(f"average: {np.mean(CorrMatrix, axis=0)}")
    #             print(f"best: {round(np.max(CorrMatrix, axis=0),3)}")
    #             print(f"when loss min: {round(CorrMatrix[0],3)}")
    #             print(f"median: {round(np.median(CorrMatrix, axis=0),3)}")
    # sys.stdout.close()
    # sys.stdout = sys.__stdout__

    # smtp_obj = smtplib.SMTP('smtp.gmail.com', 587)
    # smtp_obj.starttls()
    # smtp_obj.login('taktaksu23@gmail.com', 'atpx veir jqgp cycl')

    # body = f"Nx{XFea} についての結果ログを添付します．"
    # msg = MIMEMultipart()
    # msg['Subject'] = f"Nx{XFea}--評価結果ログ"
    # msg['To'] = 'taktaksu23@icloud.com'
    # msg['From'] = 'taktaksu23@gmail.com'
    # msg.attach(MIMEText(body))

    # file_name = f"data_save/view/{dataset_name}-Nx{XFea}.log"
    # with open(file_name, "rb") as f:
    #     attachment = MIMEApplication(f.read())

    # attachment.add_header("Content-Disposition", "attachment", filename=file_name)
    # msg.attach(attachment)
    # smtp_obj.send_message(msg)
    # smtp_obj.quit()


# x = [i for i in range(len(history_loss))]
# fig = plt.figure()

# ax1 = fig.subplots()
# ax2 = ax1.twinx()

# ax1.plot(x, history_loss, color="tab:blue", label="loss")
# ax2.plot(x, history_corr, c="tab:orange", label="corr")

# plt.title("loss and correlation")
# ax1.set_xlabel("epochs", fontsize=12)
# ax1.set_ylabel("loss", fontsize=12)
# ax2.set_ylabel("correlation", fontsize=12)

# h1, l1 = ax1.get_legend_handles_labels()
# h2, l2 = ax2.get_legend_handles_labels()
# ax1.legend(h1 + h2, l1 + l2)
# plt.show()



# plt.plot(history_loss)
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()

# plt.plot(history_corr)
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('corr')
# plt.show()


