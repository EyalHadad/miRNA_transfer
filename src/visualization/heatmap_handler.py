import pandas as pd
import csv
import os

from matplotlib import pyplot as plt

from constants import *
import pickle
import numpy as np
import copy
from statistics import mean
from sklearn import metrics
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import statistics

from constants import MODELS_CROSS_ORG_TABELS


def create_heatmaps(f_names):
    data1 = pd.read_csv(os.path.join(MODELS_CROSS_ORG_TABELS, f_names['miRNA_Net']), index_col=0)
    draw_heatmap(data=data1, img_name=f"miRNA_Net_heatmap.png", img_title='miRNA_Net')
    data2 = pd.read_csv(os.path.join(MODELS_CROSS_ORG_TABELS, f_names['xgboost']), index_col=0)
    draw_heatmap(data=data2, img_name=f"xgboost_heatmap.png", img_title='xgboost')
    draw_heatmap(data=data1 - data2, img_name=f"diff.png", img_title='Models Differences')


def draw_heatmap(data,img_name,img_title,xlabel = 'Testing Dataset',ylabel = 'Training Dataset'):
    ax = sns.heatmap(data, cmap="RdBu_r", square=True, linewidths=3, annot=True)
    plt.title(img_title, fontsize=15)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    ax.xaxis.label.set_color('purple')
    ax.title.set_color('purple')
    ax.yaxis.label.set_color('purple')
    ax.figure.tight_layout()
    plt.savefig(os.path.join(MODELS_CROSS_ORG_TABELS, img_name))
    plt.clf()