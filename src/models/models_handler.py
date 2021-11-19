import os
from constants import *
import pickle
import numpy as np
from sklearn import metrics
from datetime import datetime
import xgboost as xgb

from src.models.miRNA_transfer_subclass import api_model
from src.models.csv_handler import save_metrics


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

def create_new_model(model_type):
    if model_type == 'base':
        model = api_model(MODEL_INPUT_SHAPE)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model = xgb.XGBClassifier(kwargs=XGBS_PARAMS)  # init model
    return model


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
    return date_time, org_name, eval_dict[METRIC]


def create_res_graph(tabel_dict,org_name,model_type,trans_epochs):
    f_header = ['model'] + TRANSFER_SIZE_LIST + ['\n']
    for c in tabel_dict.keys():  # assume that c is creature
        f_name = os.path.join(MODELS_OBJECTS_GRAPHS, f"{org_name}_{c}.csv")
        with open(f_name, 'a') as file:
            if file.tell() == 0:
                file.write(','.join(str(x) for x in f_header))
            row_to_write = [f"{model_type}_{trans_epochs}"] + [str(round(tabel_dict[c][t], 2)) for t in tabel_dict[c].keys()] + ['\n']
            file.write(','.join(row_to_write))


# def create_transfer_graphs(compare_to_xgboost = True ):
#     csv_files = [x for x in os.listdir(MODELS_OBJECTS_GRAPHS) if '.csv' in x]
#     for file in csv_files:
#         data = pd.read_csv(os.path.join(MODELS_OBJECTS_GRAPHS,file),index_col=0)
#         data.dropna(how='all', axis=1, inplace=True)
#         if compare_to_xgboost:
#             data = data.loc[['base_20', 'xgboost_20']]
#         draw_transer_graph(data, file)
#
#




