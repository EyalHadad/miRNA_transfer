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
    return date_time, eval_dict['AUC']


def create_res_table(tabel_dict):
    print("--- Saving tabel results ---")
    res = pd.DataFrame(index = tabel_dict.keys(),columns=tabel_dict.keys())
    for r in tabel_dict.keys():
        for c in tabel_dict[r].keys():
            res.at[r, c] = round(tabel_dict[r][c][0],2)

    date_time = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
    res.to_csv(os.path.join(MODELS_OBJECTS_TABELS, f"{date_time}.csv"))



def create_res_graph(tabel_dict,org_name,ind,col):
        res = pd.DataFrame(index = ind,columns=col)
        for c in tabel_dict.keys():
            for t in tabel_dict[c].keys():
                res.at[c,t] = round(tabel_dict[c][t],2)
        res.T.plot()
        plt.savefig(os.path.join(MODELS_OBJECTS_GRAPHS, f"{org_name}.png"))
        plt.clf()
        res.to_csv(os.path.join(MODELS_OBJECTS_GRAPHS, f"{org_name}.csv"))

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