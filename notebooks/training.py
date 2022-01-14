from datetime import datetime
from src.models.csv_handler import save_intra_dataset
from src.models.training.xgboos_trainer import XgboostTrainObj
from src.models.training.GaussianNB_trainer import NbTrainObj
from src.models.training.RF_trainer import RfTrainObj
from src.models.training.Base_trainer import BaseTrainObj
from src.data.data_handler import *


def run_training(model_type, training_dict, models_f, train_type):
    intra_dataset_res_dict = {}
    for model_name, datasets in training_dict.items():
        if model_type == 'base':
            train_obj = BaseTrainObj(model_name)
        elif model_type == 'xgboost':
            train_obj = XgboostTrainObj(model_name)
        else:
            print("No such model type")
            return
        train_obj.train_model(models_f, model_name, datasets)
        # train_obj.model_explain()
        model_name, auc = train_obj.evaluate_model(model_type, models_f, model_name, datasets)
        intra_dataset_res_dict[model_name] = auc

    save_intra_dataset(intra_dataset_res_dict, f"{model_type}_{train_type}")


if __name__ == '__main__':
    dict_4vs4 = {"4vs4_models": VS4_REG_DICT, "4vs4_hu_models": VS4_REG_DICT_NO_HUMAN,
                 "4vs4_mu_models": VS4_REG_DICT_NO_MOUSE, "4vs4_mu_hu_models": VS4_REG_DICT_NO_BOTH}
    for k, v in PART_2_DICT.items():
        my_time = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
        models_folder_name = f"{k}_f{my_time}"
        os.mkdir(os.path.join(MODELS_OBJECTS_PATH, models_folder_name))
        run_training('base', v, models_folder_name, k)
        # run_training('xgboost', v, models_folder_name, k)
