import pandas as pd
import csv
import os
from constants import *
import pickle
import numpy as np
import copy
from sklearn import metrics
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from src.models.miRNA_transfer_subclass import api_model


def load_trained_model(model_type,org_name):
    if model_type == 'base':
        model = api_model(MODEL_INPUT_SHAPE)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        load_weights_path = os.path.join(MODELS_OBJECTS_PATH, f"{org_name}/")
        model.load_weights(load_weights_path)
    else:
        model_name = os.path.join(MODELS_OBJECTS_PATH, 'Xgboost_{0}.dat'.format(org_name))
        model = xgb.XGBClassifier(kwargs=XGBS_PARAMS)  # init model
        model.load_model(model_name)

    return model





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


def save_cross_org_table(tabel_dict, model_type):
    print("--- Saving tabel results ---")
    df = pd.read_csv(os.path.join(MODELS_INTRA_TABELS, f"{model_type}.csv"))
    in_res_dict = dict(zip(df[df.columns[0]], df[df.columns[1]]))
    res = pd.DataFrame(index = tabel_dict.keys(),columns=tabel_dict.keys())
    for r in tabel_dict.keys():
        res.at[r, r] = round(in_res_dict[r],2)
        for c in tabel_dict[r].keys():
            res.at[r, c] = round(tabel_dict[r][c][0],2)

    date_time = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
    res.to_csv(os.path.join(MODELS_CROSS_ORG_TABELS, f"{model_type}_{date_time}.csv"))


def create_res_graph(tabel_dict,org_name,model_type,trans_epochs):
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

def save_intra_dataset(table_dict, model_type):
    data=pd.DataFrame.from_dict(table_dict, orient='index')
    f_name = os.path.join(MODELS_INTRA_TABELS, f"{model_type}.csv")
    data.to_csv(f_name)


def save_transfer_table(table_dict, model_type, trans_epochs):
    transfer_dict = create_empty_species_dict()

    for src_org in table_dict.keys():
        for dst_org in table_dict.keys():
            if dst_org[:-1] in transfer_dict[src_org[:-1]]:
                for s, val in table_dict[src_org][dst_org].items():
                    transfer_dict[src_org[:-1]][dst_org[:-1]][s].append(val)
    f_name = os.path.join(MODELS_OBJECTS_TRANSFER_TABLES, f"{model_type}_{trans_epochs}_transfer.csv")
    with open(f_name, 'w') as f:
        f_header = ['src_org', 'dst_org'] + TRANSFER_SIZE_LIST + ['\n']
        f.write(','.join(str(x) for x in f_header))
        for s_org, v in transfer_dict.items():
            for d_org, t_values in transfer_dict[s_org].items():
                t_values = []
                for t_size, values in transfer_dict[s_org][d_org].items():
                    l_str = [str(round(x, 2)) for x in values]
                    t_values.append(":".join(l_str))

                row_to_write = [s_org, d_org] + [val for val in t_values] + ['\n']
                f.write(','.join(row_to_write))


def create_transfer_graphs(compare_to_xgboost = True ):
    csv_files = [x for x in os.listdir(MODELS_OBJECTS_GRAPHS) if '.csv' in x]
    for file in csv_files:
        data = pd.read_csv(os.path.join(MODELS_OBJECTS_GRAPHS,file),index_col=0)
        data.dropna(how='all', axis=1, inplace=True)
        if compare_to_xgboost:
            data = data.loc[['base_20', 'xgboost_20']]
        draw_transer_graph(data, file)


def draw_transer_graph(data, file):#TODO after calculate std
    sns.set_theme(style="darkgrid")
    xgboost_label = pd.read_csv(os.path.join(MODELS_STD_PATH,'std_xgboost.csv'),header=None,names=['model', 'value'])
    mirna_net_label = pd.read_csv(os.path.join(MODELS_STD_PATH,'std_miRNA_Net.csv'),header=None,names=['model', 'value'])
    ax = sns.lineplot(data=data.T, markers=True, dashes=False, linewidth=2.5)
    plt.annotate("Point 1", (1,data.loc['xgboost_20','100']))
    # for item, color in zip(data.groupby('0'),['r','b','g']):
    #     # item[1] is a grouped data frame
    #     for x, y, m in item[1][['100', '200','700']].values:
    #         ax.text(x, y, f'{m:.2f}', color=color)
    #

    plt.title(file.split('.')[0], fontsize=15)
    plt.xlabel('#Target observations', fontsize=10)
    plt.ylabel('AUC', fontsize=10)
    ax.title.set_color('purple')
    ax.figure.tight_layout()
    plt.savefig(os.path.join(MODELS_OBJECTS_GRAPHS, f"{file.split('.')[0]}.png"))
    plt.clf()




