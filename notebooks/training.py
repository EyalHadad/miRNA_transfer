from src.models.training.xgboos_trainer import XgboostTrainObj
from src.models.training.GaussianNB_trainer import NbTrainObj
from src.models.training.RF_trainer import RfTrainObj
from src.models.training.Base_trainer import BaseTrainObj
from src.data.data_handler import *


def run_training():
    intra_dataset_res_dict = {}
    dataset_list = DATASETS
    # dataset_list = ['worm1']
    for model_type in [4]:
        for data in dataset_list:
            if model_type == 1:
                train_obj = XgboostTrainObj(data, 'Xgboost')
            elif model_type == 2:
                train_obj = NbTrainObj(data, 'GaussianNB')
            elif model_type == 3:
                train_obj = RfTrainObj(data, 'Random forest')
            else:
                train_obj = BaseTrainObj(data, 'Base')

            train_obj.train_model()
            # train_obj.model_explain()
            model_name,auc = train_obj.evaluate_model()
            intra_dataset_res_dict[model_name] = auc
    return intra_dataset_res_dict


if __name__ == '__main__':
    run_training()
