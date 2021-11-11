import pandas as pd
import csv
import os
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

from constants import MODELS_CROSS_ORG_TABELS, MODELS_STD_PATH


def calculate_std(f_names):#TODO read right tabel
    data1 = pd.read_csv(os.path.join(MODELS_CROSS_ORG_TABELS, f_names['miRNA_Net']), index_col=0)
    save_std(data=data1,output_f_name=f"std_miRNA_Net.csv")
    data2 = pd.read_csv(os.path.join(MODELS_CROSS_ORG_TABELS, f_names['xgboost']), index_col=0)
    save_std(data=data2, output_f_name=f"std_xgboost.csv")


def save_std(data, output_f_name):
    res_dict = defaultdict(list)
    std_dict = {'cow_worm':[('cow1','worm1'),('cow1','worm2')]
                ,'cow_human':[('cow1','human1'),('cow1','human2'),('cow1','human3')]
                ,'cow_mouse':[('cow1','mouse1'),('cow1','mouse2')]
                ,'worm_cow':[('worm1','cow1'),('worm2','cow1')]
                ,'worm_human':[('worm1','human1'),('worm1','human2'),('worm1','human3'),('worm2','human1'),('worm2','human2'),('worm2','human3')]
                ,'worm_mouse':[('worm1','mouse1'),('worm1','mouse2'),('worm2','mouse1'),('worm2','mouse2')]
                , 'human_cow': [('human1','cow1'), ('human2','cow1'), ('human3','cow1')]
                , 'human_worm': [('human1','worm1'),('human2','worm1'),('human3','worm1'),('human1','worm2'),('human2','worm2'),('human3','worm2')]
                , 'human_mouse': [('human1', 'mouse1'), ('human2', 'mouse1'), ('human3', 'mouse1'), ('human1', 'mouse2'),('human2', 'mouse2'), ('human3', 'mouse2')]
                , 'mouse_cow': [('mouse1', 'cow1'), ('mouse2', 'cow1')]
                , 'mouse_human': [('mouse1', 'human1'), ('mouse1', 'human2'), ('mouse1', 'human3'), ('mouse2', 'human1'),('mouse2', 'human2'), ('mouse2', 'human3')]
                , 'mouse_worm': [('mouse1','worm1'), ('mouse2','worm1'), ('mouse1','worm2'), ('mouse2','worm2')]
                }
    for k,v in std_dict.items():
        for elm in std_dict[k]:
            res_dict[k].append(data.loc[elm[0],elm[1]])
    with open(os.path.join(MODELS_STD_PATH,output_f_name),'w') as f:
        for k,v in res_dict.items():
            f.write("{0},{1}\n".format(k,statistics.stdev(v)))