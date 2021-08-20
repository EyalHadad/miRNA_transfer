import csv
import os
from constants import *
import pickle
from datetime import datetime


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


def save_pkl_model(model, pkl_filename):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
