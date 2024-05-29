#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import os
import atexit
import matplotlib as mpl
import time
import pandas as pd
from scipy import stats
import japanize_matplotlib

"""ブラウザでmatplotlib表示"""


def is_under_ssh_connection():
    # The environment variable `SSH_CONNECTION` exists only in the SSH session.
    return "SSH_CONNECTION" in os.environ.keys()


def use_WebAgg(port=8000, port_retries=50):
    """use WebAgg for matplotlib backend."""
    current_backend = mpl.get_backend()
    current_webagg_configs = {
        "port": mpl.rcParams["webagg.port"],
        "port_retries": mpl.rcParams["webagg.port_retries"],
        "open_in_browser": mpl.rcParams["webagg.open_in_browser"],
    }

    def reset():
        mpl.use(current_backend)
        mpl.rc("webagg", **current_webagg_configs)

    mpl.use("WebAgg")
    mpl.rc(
        "webagg",
        **{"port": port, "port_retries": port_retries, "open_in_browser": False},
    )
    atexit.register(reset)


if is_under_ssh_connection():
    use_WebAgg(port=8000, port_retries=50)
""""""


def get_95(df):
    a = 0.95  # 信頼水準
    d = len(df) - 1  # 自由度
    m = df.mean()  # 標本平均
    s = stats.sem(df)  # 標準誤差
    target_95per_section = stats.t.interval(a, d, m, s)
    target_95per = (target_95per_section[1] - target_95per_section[0]) / 2
    return target_95per


"""choice"""
f = 9  # feature
# p = 8 #target
n = 4000  # samples
error = []
y = []
for p in range(1, 9):
    df = pd.DataFrame()
    CorrSummary = np.loadtxt(
        f"data_save/evaluation/4000x9-mimic-random/{p}-columns/CorrSummary.txt"
    )
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


figure_width = 11.7  # A4 paper width in inches
# figure_height = figure_width / ((1 + np.sqrt(5.0)) / 2.0)  # Golden ratio
figure_height = (figure_width / 4) * 3

linewidth = 2.0
capsize=10
markersize=12

plt.rcParams["font.size"] = 38
# plt.rcParams["font.family"] = ["Fira Sans", "Open Sans"]
plt.rcParams["font.weight"] = 400
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["pdf.fonttype"] = 42

fig, ax = plt.subplots(figsize=(figure_width, figure_height))

ax.errorbar(
    [1, 2, 3, 4, 5, 6, 7, 8],
    y,
    yerr=error,
    marker="o",
    capthick=linewidth,
    capsize=capsize,
    markersize=markersize,
    color="black",
    label="再構成したデータ",
    markeredgecolor="black",
    markerfacecolor="white",
    markeredgewidth=linewidth,
    linewidth=linewidth,
)
ax.hlines(
    0.009,
    1,
    8,
    color="black",
    linestyles="dashed",
    label="ランダムに生成したデータ",
    linewidth=1.5,
)
ax.legend(frameon=False, borderaxespad=0, labelspacing=0.1)
ax.set_xlabel(r"$X_{\rho}^{O}$ ($|\mathcal{F}_{\rho}|$) 内の特徴数")
ax.set_ylabel("PCC スコア")
ax.set_ylim(0, 1.1)
ax.set_xlim(0.9, 8.1)
ax.set_yticks(np.arange(0, 1.1, 0.2))
ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])

# 直下にpdf保存
fig.set_tight_layout(True)
plt.savefig("data_save/numFeaturesAndPCC.pdf")
plt.show()
