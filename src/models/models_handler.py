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


def save_feature_importance_res(row_desc, f_importance_list,type_name):
    f_path = os.path.join(MODELS_FEATURE_IMPORTANCE, 'models_feature_importance_{0}.csv'.format(type_name))
    f_importance_str = ",".join(["{0}:{1}".format(x[0], x[1]) for x in f_importance_list[:10]])
    date_time = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
    with open(f_path, 'a') as file:
        file.write('{0},{1},{2}'.format(row_desc, date_time, f_importance_str))
        file.write("\n")


def save_metrics(eval_dict):
    f_path = os.path.join(MODELS_PATH, 'models_evaluation.csv')
    with open(f_path, 'a') as file:
        writer = csv.DictWriter(file, eval_dict.keys(), delimiter=',', lineterminator='\n')
        if file.tell() == 0:
            writer.writeheader()  # file doesn't exist yet, write a header
        writer.writerow(eval_dict)


def save_pkl_model(model,pkl_filename):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


def cat_or_pad(x, length):
    if len(x)>length:
        return x[:length]
    else:
        return x + ['N'] * (length - len(x))


def create_sequence(miRNA, mRNA):
    my_map = {'A': 0, 'C': 1,'G': 2, 'U': 3,'T': 3,'N': 4}
    miRNA = cat_or_pad(list(miRNA),30)
    mRNA = cat_or_pad(list(mRNA),100)
    total_res = [my_map[x] for x in miRNA+mRNA]
    return total_res


def create_evaluation_dict(t_model_name, org_name, pred, y):
    model_name = '{0}_{1}'.format(t_model_name, org_name)
    date_time = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
    print(f" There are {np.sum(np.isnan(pred))} nan predictions")
    np.nan_to_num(pred,copy=False)
    print(f" After filling 0 instead of nan there are {np.sum(np.isnan(pred))} nan predictions")
    eval_dict = {'Model': model_name, 'Date': date_time, 'ACC': metrics.accuracy_score(y, np.round(pred))}
    eval_dict['FPR'], eval_dict['TPR'], thresholds = metrics.roc_curve(y, pred)
    eval_dict['AUC'] = metrics.auc(eval_dict['FPR'], eval_dict['TPR'])
    eval_dict['MCC'] = metrics.matthews_corrcoef(y, np.round(pred))
    eval_dict['F1_score'] = metrics.f1_score(y, np.round(pred))
    save_metrics(eval_dict)
    return date_time, org_name, eval_dict['AUC']


def create_res_table(tabel_dict,in_res_dict):
    print("--- Saving tabel results ---")
    res = pd.DataFrame(index = tabel_dict.keys(),columns=tabel_dict.keys())
    for r in tabel_dict.keys():
        res.at[r, r] = round(in_res_dict[r],2)
        for c in tabel_dict[r].keys():
            res.at[r, c] = round(tabel_dict[r][c][0],2)

    date_time = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
    res.to_csv(os.path.join(MODELS_OBJECTS_TABELS, f"{date_time}.csv"))



def create_res_graph(tabel_dict,org_name,col,model_type,trans_epochs):
    f_header = ['model'] + TRANSFER_SIZE_LIST + ['\n']
    for c in tabel_dict.keys():  # assume that c is creature
        f_name = os.path.join(MODELS_OBJECTS_GRAPHS, f"{org_name}_{c}.csv")
        with open(f_name, 'a') as file:
            if file.tell() == 0:
                file.write(','.join(str(x) for x in f_header))
            row_to_write = [f"{model_type}_{trans_epochs}"] + [str(round(tabel_dict[c][t], 2)) for t in tabel_dict[c].keys()] + ['\n']
            file.write(','.join(row_to_write))

def create_empty_species_dict():
    di = {}
    for a in SPECIES:
        di[a] = dict()
    for a in SPECIES:
        rest = copy.deepcopy(SPECIES)
        rest.remove(a)
        for r in rest:
            di[a][r] = {}
            for s in TRANSFER_SIZE_LIST:
                di[a][r][s] = []
    return di


def create_species_dict(table_dict):
    new_dict = create_empty_species_dict()
    for src_org in table_dict.keys():
        for dst_org in table_dict.keys():
            if dst_org[:-1] in new_dict[src_org[:-1]]:
                for s, val in table_dict[src_org][dst_org].items():
                    new_dict[src_org[:-1]][dst_org[:-1]][s].append(val)
    for s_org in new_dict.keys():
        for d_org in new_dict[s_org].keys():
            for s,val in new_dict[s_org][d_org].items():
                org_list = new_dict[s_org][d_org][s]
                new_dict[s_org][d_org][s] = round(mean(org_list),2)
    return new_dict


def create_transfer_graphs(compare_to_xgboost = True ):
    csv_files = [x for x in os.listdir(MODELS_OBJECTS_GRAPHS) if '.csv' in x]
    for file in csv_files:
        data = pd.read_csv(os.path.join(MODELS_OBJECTS_GRAPHS,file),index_col=0)
        data.dropna(how='all', axis=1, inplace=True)
        if compare_to_xgboost:
            data = data.loc[['base_20', 'xgboost_20']]
        data.T.plot(title=file.split('.')[0])
        plt.savefig(os.path.join(MODELS_OBJECTS_GRAPHS, f"{file.split('.')[0]}.png"))
        plt.clf()


def draw_heatmap(data,img_name,img_title,xlabel = 'Testing Dataset',ylabel = 'Training Dataset'):
    ax = sns.heatmap(data, cmap="RdBu_r", square=True, linewidths=3, annot=True)
    plt.title(img_title, fontsize=15)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    ax.xaxis.label.set_color('purple')
    ax.title.set_color('purple')
    ax.yaxis.label.set_color('purple')
    ax.figure.tight_layout()
    plt.savefig(os.path.join(MODELS_OBJECTS_TABELS, img_name))
    plt.clf()


def create_heatmaps(f_names):
    data1 = pd.read_csv(os.path.join(MODELS_OBJECTS_TABELS, f_names['miRNA_Net']), index_col=0)
    draw_heatmap(data=data1, img_name=f"miRNA_Net_heatmap.png", img_title='miRNA_Net')
    data2 = pd.read_csv(os.path.join(MODELS_OBJECTS_TABELS, f_names['xgboost']), index_col=0)
    draw_heatmap(data=data2, img_name=f"xgboost_heatmap.png", img_title='xgboost')
    draw_heatmap(data=data1 - data2, img_name=f"diff.png", img_title='Models Differences')


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



def calculate_std(f_names):
    data1 = pd.read_csv(os.path.join(MODELS_OBJECTS_TABELS, f_names['miRNA_Net']), index_col=0)
    save_std(data=data1,output_f_name=f"std_miRNA_Net.csv")
    data2 = pd.read_csv(os.path.join(MODELS_OBJECTS_TABELS, f_names['xgboost']), index_col=0)
    save_std(data=data2, output_f_name=f"std_xgboost.csv")


