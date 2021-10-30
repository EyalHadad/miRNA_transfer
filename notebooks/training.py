from src.models.training.xgboos_trainer import XgboostTrainObj
from src.models.training.GaussianNB_trainer import NbTrainObj
from src.models.training.RF_trainer import RfTrainObj
from src.models.training.Base_trainer import BaseTrainObj
from src.data.data_handler import *


def run_training(model_type='base'):
    intra_dataset_res_dict = {}
    dataset_list = DATASETS
    # dataset_list = ['worm1']
    for data in dataset_list:
        if model_type == 'base':
            train_obj = BaseTrainObj(data, 'Base')
        elif model_type == 'xgboost':
            train_obj = XgboostTrainObj(data, 'Xgboost')
        else:
            print("No such model type")
            return
        train_obj.train_model()
        # train_obj.model_explain()
        model_name,auc = train_obj.evaluate_model()
        intra_dataset_res_dict[model_name] = auc
    return intra_dataset_res_dict


if __name__ == '__main__':
    run_training()
