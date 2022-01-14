from src.models.csv_handler import save_intra_dataset
from src.models.training.xgboos_trainer import XgboostTrainObj
from src.models.training.Base_trainer import BaseTrainObj
from src.data.data_handler import *
from src.models.csv_handler import *
import copy


def create_cross_org_tables(iteration_dict, models_f, training_type,metric='ACC'):
    xgb_cross_org_dict,base_cross_org_dict = {},{}
    for src_model_name, src_dataset_list in iteration_dict.items():
        xgb_cross_org_dict[src_model_name] = {}
        base_cross_org_dict[src_model_name] = {}
        base_obj = BaseTrainObj(src_model_name)
        xgb_obj = XgboostTrainObj(src_model_name)
        for dst_model_name, dst_dataset_list in iteration_dict.items():
            # xgb_cross_org_dict[src_model_name] = {}
            # base_cross_org_dict[src_model_name] = {}
            m_name,score = base_obj.evaluate_model('base', models_f, src_model_name, dst_dataset_list,True,metric)
            base_cross_org_dict[src_model_name][dst_model_name] = score
            # m_name,score = xgb_obj.evaluate_model('xgboost', models_f, src_model_name, dst_dataset_list,True,metric)
            # xgb_cross_org_dict[src_model_name][dst_model_name] = score

    df_base = pd.DataFrame.from_dict(base_cross_org_dict).round(2).T
    # df_xgb = pd.DataFrame.from_dict(xgb_cross_org_dict).round(2).T
    time_var = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
    df_base.to_csv(os.path.join(MODELS_CROSS_ORG_TABELS, f"base_{metric}_{training_type}_{time_var}.csv"))
    # df_xgb.to_csv(os.path.join(MODELS_CROSS_ORG_TABELS, f"xgboost_{metric}_{training_type}_{time_var}.csv"))




if __name__ == '__main__':

    for k, v in PART_2_DICT.items():
        models_folder_name = os.path.join(MODELS_OBJECTS_PATH, k)
        create_cross_org_tables(v, models_folder_name,k)


    # models_folder_name = "dataset_models_" + datetime.now().strftime("%d_%m_%Y %H_%M_%S")
    # models_folder_name = os.path.join(MODELS_OBJECTS_PATH, "dataset_models")
    # create_cross_org_tables(PART_2_DICT., models_folder_name)


    # create_cross_org_tables(TRAIN_DICT_REG, models_folder_name,"8X8")
