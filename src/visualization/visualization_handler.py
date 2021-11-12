import statistics
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt

from constants import *
from constants import MODELS_STATISTICS_PATH


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
    plt.savefig(os.path.join(MODELS_GRAPHS_HEATMAP, img_name))
    plt.clf()


def get_statistic_str(row, action):
    str_list = [row[str(x)].split(":") for x in TRANSFER_SIZE_LIST]
    num_list = [[float(x) for x in lst] for lst in str_list]
    if action=='std':
        num_list = [round(statistics.stdev(v),2) for v in num_list]
    elif action=='avg':
        num_list = [round(statistics.mean(v), 2) for v in num_list]
    r_list = ','.join([str(x) for x in num_list])
    return r_list


def save_statistic_file(model_type, action):
    data = pd.read_csv(os.path.join(MODELS_OBJECTS_TRANSFER_TABLES, f"{model_type}_transfer.csv"))
    out_f = os.path.join(MODELS_STATISTICS_PATH, f"{model_type}_{action}.csv")
    with open(out_f, 'w') as f:
        f_header = ['model'] + TRANSFER_SIZE_LIST + ['\n']
        f.write(','.join(str(x) for x in f_header))
        data.apply(lambda row: f.write("{0}_{1},{2}\n".format(row['src_org'], row['dst_org'], get_statistic_str(row, action))), axis=1)


def file_exists(model,file_type):
    f_path = os.path.join(MODELS_STATISTICS_PATH, f"{model}_{file_type}.csv")
    return os.path.exists(f_path)

