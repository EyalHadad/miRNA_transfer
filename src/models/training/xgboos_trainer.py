from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import shap
import numpy as np
import os
from constants import *
from src.models.models_handler import save_metrics
from time import gmtime, strftime
from sklearn.model_selection import train_test_split
from sklearn import metrics

class XgboostTrainObj:
    x = None
    xval = None
    y = None
    yval = None
    model = None
    org_name=None


    def __init__(self,org_name):
        self.org_name = org_name


    def train_model(self):
        train = pd.read_csv(os.path.join(PROCESSED_TRAIN_PATH, "{0}_train.csv".format(self.org_name)),index_col=False)
        print("---Train data was loaded---\n")
        print("training data shape:", train.shape)
        y = train['label']
        x = train.drop(['label'], axis=1)
        x = np.asarray(x).astype('float32')
        self.x, self.xval, self.y, self.yval = train_test_split(x, y, test_size=0.1, random_state=42)
        print("---Start training---\n")
        eval_s = [(self.x, self.y), (self.xval, self.yval)]
        print("---Start training---\n")
        self.model = xgb.XGBClassifier().fit(self.x, self.y, eval_metric=["error", "logloss"], eval_set=eval_s)

        self.plot_learning_curves()
        model_name = os.path.join(MODELS_OBJECTS_PATH,
                                  'xgboost_{0}_{1}.json'.format(self.org_name, strftime("%Y-%m-%d", gmtime())))
        self.model.save_model(model_name)

        # plt.figure(figsize=(16, 12))
        # xgb.plot_importance(self.model)
        # plt.show()



    def plot_learning_curves(self):
        results = self.model.evals_result()
        epochs = len(results['validation_0']['error'])
        x_axis = range(0, epochs)
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
        ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
        ax.legend()
        plt.ylabel('Log Loss')
        plt.title('XGBoost Log Loss')
        plt.savefig(os.path.join(MODELS_OUTPUT_PATH, 'XGBoost {0} Log Loss.png'.format(self.org_name)))
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.plot(x_axis, results['validation_0']['error'], label='Train')
        ax.plot(x_axis, results['validation_1']['error'], label='Test')
        ax.legend()
        plt.ylabel('Classification Error')
        plt.title('XGBoost Classification Error')
        plt.savefig(os.path.join(MODELS_OUTPUT_PATH, 'XGBoost {0} Classification Error.png'.format(self.org_name)))

    def explain_model(self):
        print("---Explain model---\n")
        explainer = shap.Explainer(self.model)
        shap_values = explainer(self.x)
        shap.plots.waterfall(shap_values[0], show=False)
        plt.savefig(os.path.join(MODELS_OUTPUT_PATH, 'xgboost_{0}_waterfall.png'.format(self.org_name)))
        shap.summary_plot(shap_values, self.x, plot_type="bar", show=False)
        plt.savefig(os.path.join(MODELS_OUTPUT_PATH, 'xgboost_{0}_bar.png'.format(self.org_name)))
        print("---Shap plots were saved---\n")

    def evaluate_model(self):
        test = pd.read_csv(os.path.join(PROCESSED_TEST_PATH, "{0}_test.csv".format(self.org_name)),index_col=False)
        y = test['label']
        x = test.drop(['label'], axis=1)
        x = np.asarray(x).astype('float32')
        pred = self.model.predict(x)
        model_name =  'xgboost_{0}'.format(self.org_name)
        date_time = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
        eval_dict = {'Model':model_name, 'Date': date_time, 'ACC': metrics.accuracy_score(y, pred)}
        eval_dict['FPR'], eval_dict['TPR'], thresholds = metrics.roc_curve(y, pred, pos_label=2)
        eval_dict['AUC'] = metrics.auc(eval_dict['FPR'], eval_dict['TPR'])
        eval_dict['MCC'] = metrics.matthews_corrcoef(y, pred)
        eval_dict['F1_score'] = metrics.f1_score(y, pred)
        save_metrics(eval_dict)
        pred_res = pd.DataFrame(zip(pred, y), columns=['pred', 'y'])
        pred_res.to_csv(os.path.join(MODELS_PREDICTION_PATH, "pred_{0}_{1}.csv".format(model_name,date_time)),index=False)

