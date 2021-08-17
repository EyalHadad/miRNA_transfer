from src.models.training.xgboos_trainer import XgboostTrainObj

if __name__ == '__main__':
    # dataset_list = ['cow', 'human1', 'human2', 'human3', 'worm1', 'worm2', 'mouse1', 'mouse2']
    dataset_list = ['cow']
    for data in dataset_list:
        train_obj = XgboostTrainObj(data)
        train_obj.train_model()
        train_obj.explain_model()
        train_obj.evaluate_model()

