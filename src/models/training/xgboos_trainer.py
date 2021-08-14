import matplotlib.pyplot as plt
import pandas as pd
import xgboost
import shap
import numpy as np
import os
from constants import *
from time import gmtime, strftime
from sklearn.model_selection import train_test_split


class XgboostTrainObj:
    x = None
    xval = None
    y = None
    yval = None
    model = None

    def train_model(self):
        train = pd.read_csv(os.path.join(PROCESSED_TRAIN_PATH, "train.csv"), index_col=False)
        print("---Train data was loaded---\n")
        print("training data shape:", train.shape)
        y = train['label']
        x = train.drop(['label'], axis=1)
        x = np.asarray(x).astype('float32')
        self.x, self.xval, self.y, self.yval = train_test_split(x, y, test_size=0.1, random_state=42)
        print("---Start training---\n")
        eval_s = [(self.x, self.y), (self.xval, self.yval)]
        self.model = xgboost.XGBClassifier().fit(self.x, self.y, eval_metric=["error", "logloss"], eval_set=eval_s)
        self.plot_learning_curves()
        i = 8
        model_name = os.path.join(MODELS_OBJECTS_PATH, 'xgboost_{0}.json'.format(strftime("%Y-%m-%d", gmtime())))
        self.model.save_model(model_name)

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
        plt.savefig(os.path.join(MODELS_OUTPUT_PATH, 'XGBoost Log Loss.png'))
        # plot classification error
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.plot(x_axis, results['validation_0']['error'], label='Train')
        ax.plot(x_axis, results['validation_1']['error'], label='Test')
        ax.legend()
        plt.ylabel('Classification Error')
        plt.title('XGBoost Classification Error')
        plt.savefig(os.path.join(MODELS_OUTPUT_PATH, 'XGBoost Classification Error.png'))

    def explain_model(self):
        print("---Explain model---\n")
        explainer = shap.Explainer(self.model)
        shap_values = explainer(self.x)
        shap.plots.waterfall(shap_values[0], show=False)
        plt.savefig(os.path.join(MODELS_OUTPUT_PATH, 'xgboost_shap_waterfall.png'))
        shap.summary_plot(shap_values, self.x, plot_type="bar", show=False)
        plt.savefig(os.path.join(MODELS_OUTPUT_PATH, 'xgboost_shap_bar.png'))
        print("---Shap plots were saved---\n")

    def evaluate_model(self):
        # TODO
        pass
