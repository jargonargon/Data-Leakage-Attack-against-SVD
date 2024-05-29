import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from livelossplot import PlotLosses
import os
import atexit
import matplotlib as mpl
import time
import pandas as pd
from scipy import stats
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42


def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

'''ブラウザでmatplotlib表示'''
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
''''''

def get_95(df):
    a = 0.95 #信頼水準
    d = len(df)-1 #自由度
    m = df.mean() #標本平均
    s = stats.sem(df) #標準誤差
    target_95per_section = stats.t.interval(a,d,m,s)
    target_95per = (target_95per_section[1] - target_95per_section[0])/2
    return target_95per 



'''choice'''
f = 9 #feature
# p = 8 #target
n = 4000 #samples
error = []
y = []
for p in range(1,9):
    df = pd.DataFrame()
    CorrSummary = np.loadtxt(f"data_save/evaluation/{n}x{f}-mimic-random/{p}-columns/CorrSummary.txt")
    if p != 1:
        CorrSummary = np.mean(CorrSummary, axis=1)
    for iter in range(10):
        lst = [[p, CorrSummary[iter]]]
        column = ["Feature", "Corr"]
        df1 = pd.DataFrame(data=lst, columns=column)
        df = pd.concat([df, df1], ignore_index=False)
    conf_95 = get_95(df["Corr"])
    error.append(conf_95)
    y.append(np.mean(CorrSummary))


fig, ax = plt.subplots()
ax.plot([1,2,3,4,5,6,7,8], y, color='black')
ax.errorbar([1,2,3,4,5,6,7,8], y, yerr=error, marker='o', capthick=1, capsize=5, lw=1, color='black', label="reconstructed data")
plt.hlines(0.009, 1, 8, color='black', linestyles='dotted', label="randomly generated data")
plt.legend(fontsize=12)
plt.xlabel("Features to be reconstructed", fontsize="12")
plt.ylabel("PCC score", fontsize="12")
plt.ylim(0,1.1)

# 直下にpdf保存
plt.savefig('numFeaturesAndPCC.pdf')
plt.show()