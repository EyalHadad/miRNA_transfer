import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from constants import *
from constants import MODELS_OBJECTS_GRAPHS
from src.visualization.visualization_handler import save_statistic_file, file_exists
from collections import defaultdict


def create_transfer_graphs(model_list):
    graph_dict = dict()
    std_dict = dict()
    print("loading transfer data results")
    for model in model_list:
        if not file_exists(model, 'avg'):
            save_statistic_file(model, 'avg')
        f_name = os.path.join(MODELS_STATISTICS_PATH, f"{model}_avg.csv")
        graph_dict[model] = pd.read_csv(f_name, index_col=['model']).iloc[:, :-1]
        if not file_exists(model, 'std'):
            save_statistic_file(model, 'std')
        f_name = os.path.join(MODELS_STATISTICS_PATH, f"{model}_std.csv")
        std_dict[model] = pd.read_csv(f_name, index_col=['model']).iloc[:, :-1]

    print("Drawing transfer graphs")
    transfer_model_index = graph_dict[list(graph_dict.keys())[0]].index
    for ind in transfer_model_index:
        plot_df = pd.DataFrame()
        for k, trans_df in graph_dict.items():
            plot_df = plot_df.append(pd.Series(data=trans_df.loc[ind], name=k))
        draw_transfer_graph(plot_df, ind, std_dict)


def draw_transfer_graph(data, org_names, std_dict):
    sns.set_theme(style="darkgrid")
    labels =[x.replace("base_", "miRNA_NET_") for x in data.index]
    offset_dict = {'base_20': 5, 'xgboost_20': -10}
    offset_dict = defaultdict(lambda: -10, offset_dict)
    ax = sns.lineplot(data=data.T, dashes=False, linewidth=2.5)
    ax.legend(labels)
    for model_t, row in data.iterrows():
        for transfer_size, value in data.items():
            res = std_dict[model_t].loc[org_names][transfer_size]
            ax.annotate(text=res, xy=(transfer_size, data.loc[model_t][transfer_size]), xycoords='data',bbox=dict(boxstyle='round', fc='0.9',pad=0.3),
                        fontsize=7, xytext=(-5, offset_dict[model_t]), textcoords='offset points')
    plt.title(f"{org_names}", fontsize=15)
    plt.xlabel('#Target observations', fontsize=10)
    plt.ylabel('AUC', fontsize=10)
    ax.title.set_color('purple')
    ax.figure.tight_layout()
    plt.savefig(os.path.join(MODELS_OBJECTS_GRAPHS, f"{org_names}.png"))
    plt.clf()