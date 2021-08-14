from src.models.training.xgboos_trainer import XgboostTrainObj

if __name__ == '__main__':

    train_obj = XgboostTrainObj()
    train_obj.train_model()
    train_obj.explain_model()

