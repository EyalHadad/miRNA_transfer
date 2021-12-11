from src.models.csv_handler import save_intra_dataset
from src.models.training.xgboos_trainer import XgboostTrainObj
from src.models.training.Base_trainer import BaseTrainObj
from src.data.data_handler import *


def all_against_one_training(model_type='base'):
    intra_dataset_res_dict = {}
    dataset_list = SPECIES.copy()
    for data in dataset_list:
        if model_type == 'base':
            train_obj = BaseTrainObj(data, 'Base')
        elif model_type == 'xgboost':
            train_obj = XgboostTrainObj(data, 'Xgboost')
        else:
            print("No such model type")
            return
        train_obj.train_model(True)
        # train_obj.model_explain()
        model_name,auc = train_obj.evaluate_model(model_type,True)
        intra_dataset_res_dict[model_name] = auc

    save_intra_dataset(intra_dataset_res_dict,model_type,True)


if __name__ == '__main__':
    # all_against_one_training('xgboost')
    all_against_one_training('base')
