from src.models.training.xgboos_trainer import XgboostTrainObj
from src.models.training.GaussianNB_trainer import NbTrainObj
from src.models.training.RF_trainer import RfTrainObj

if __name__ == '__main__':
    # dataset_list = ['cow', 'human1', 'human2', 'human3', 'worm1', 'worm2', 'mouse1', 'mouse2']
    dataset_list = ['cow']
    for model_type in [1]:
        for data in dataset_list:
            if model_type == 1:
                train_obj = XgboostTrainObj(data, 'Xgboost')
            elif model_type == 2:
                train_obj = NbTrainObj(data, 'GaussianNB')
            else:
                train_obj = RfTrainObj(data, 'Random forest')

            train_obj.train_model()
            train_obj.model_explain()
            train_obj.evaluate_model()
