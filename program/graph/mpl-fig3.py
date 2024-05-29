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


f = 9 #feature
p = 7 #target
n = 4000 #samples

LossArr = np.empty(0)
CorrArr = np.empty(0)

for i in range(100):
    Corr = np.loadtxt(f"data_save/evaluation/{n}x{f}-mimic-random-test/{p}-columns/CorrHistory-{i}.txt")
    Loss = np.loadtxt(f"data_save/evaluation/{n}x{f}-mimic-random-test/{p}-columns/LossHistory-{i}.txt")
    minLossIndex = np.argmin(Loss)
    LossArr = np.append(LossArr, Loss[minLossIndex])
    CorrArr = np.append(CorrArr, Corr[minLossIndex])

LossArr = LossArr* (4/p)
ta_koukei_keisu = np.polyfit(LossArr, CorrArr, 1)

plt.scatter(LossArr, CorrArr, c="black")
plt.ylim(0,1)
plt.plot(LossArr, np.polyval(ta_koukei_keisu, LossArr), color='red', label='Approximation line')
print(ta_koukei_keisu)
plt.show()