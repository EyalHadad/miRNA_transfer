import csv
import os
from constants import *
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from datetime import datetime
from sklearn import metrics


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
    my_map = {'A': 0, 'C': 1,'G': 2, 'T': 3,'N': 4}
    miRNA = cat_or_pad(list(miRNA),30)
    mRNA = cat_or_pad(list(mRNA),100)
    total_res = [my_map[x] for x in miRNA+mRNA]
    return total_res


def create_evaluation_dict(t_model_name, org_name, pred, y):
    model_name = '{0}_{1}'.format(t_model_name, org_name)
    date_time = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
    eval_dict = {'Model': model_name, 'Date': date_time, 'ACC': metrics.accuracy_score(y, np.round(pred))}
    eval_dict['FPR'], eval_dict['TPR'], thresholds = metrics.roc_curve(y, pred)
    eval_dict['AUC'] = metrics.auc(eval_dict['FPR'], eval_dict['TPR'])
    eval_dict['MCC'] = metrics.matthews_corrcoef(y, np.round(pred))
    eval_dict['F1_score'] = metrics.f1_score(y, np.round(pred))
    save_metrics(eval_dict)
    return date_time, model_name