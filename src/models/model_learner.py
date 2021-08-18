from sklearn.inspection import permutation_importance
from datetime import datetime
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import shap
import numpy as np
from constants import *
from sklearn.model_selection import train_test_split
import os
from src.models.models_handler import save_metrics

class ModelLearner:
    x = None
    xval = None
    y = None
    yval = None
    model = None
    org_name=None
    feature_names = None
    model_name = None


    def __init__(self,org_name,m_name):
        self.org_name = org_name
        self.model_name = m_name



    def prep_model_training(self):
        train = pd.read_csv(os.path.join(PROCESSED_TRAIN_PATH, "{0}_train.csv".format(self.org_name)),index_col=False)
        print("---Train data was loaded---\n")
        print("training data shape:", train.shape)
        y = train['label']
        X = train.drop(['mRNA_start', 'label'], axis=1)
        self.feature_names = list(X.columns)
        X[np.isnan(X)] = 0
        self.x, self.xval, self.y, self.yval = train_test_split(X, y, test_size=0.2, random_state=42)



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
        plt.clf()


    def model_explain(self):
        print("Shap values\n")
        explainer = shap.Explainer(self.model,feature_names=self.feature_names)
        shap_values = explainer(self.x)
        shap.plots.waterfall(shap_values[0], show=False)
        plt.title('{0} {1} waterfall'.format(self.model_name,self.org_name))
        plt.savefig(os.path.join(MODELS_FEATURE_IMPORTANCE, '{0}_{1}_waterfall.png'.format(self.model_name,self.org_name)))
        plt.clf()
        shap.summary_plot(shap_values, plot_type="bar", show=False, max_display=10)
        plt.title('{0} {1} SHAP bar'.format(self.model_name,self.org_name))
        plt.savefig(os.path.join(MODELS_FEATURE_IMPORTANCE, '{0}_{1}_bar.png'.format(self.model_name,self.org_name)))
        plt.clf()
        print("---Shap plots were saved---\n")


    def calc_permutation_importance(self):

        imps = permutation_importance(self.model, self.xval, self.yval)
        importances = imps.importances_mean
        std = imps.importances_std
        indices = np.argsort(importances)[::-1]
        title = '{0} {1} f_important'.format(self.model_name, self.org_name)
        #TODO move it to handler file (plot_figure)- need to know first what argument to send
        plt.figure(figsize=(10, 7))
        plt.title(title)
        plt.bar(range(self.xval.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
        # plt.xticks(range(X_test.shape[1]), [features[indices[i]] for i in range(6)])
        # plt.xlim([-1, X_test.shape[1]])
        plt.savefig(os.path.join(MODELS_FEATURE_IMPORTANCE, '{0}_bar.png'.format(title)))
        plt.clf()


    def evaluate_model(self):
        test = pd.read_csv(os.path.join(PROCESSED_TEST_PATH, "{0}_test.csv".format(self.org_name)), index_col=False)
        y = test['label']
        x = test.drop(['mRNA_start', 'label'], axis=1)
        x = np.asarray(x).astype('float32')
        pred = self.model.predict(x)
        model_name = '{0}_{1}'.format(self.model_name, self.org_name)
        date_time = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
        eval_dict = {'Model': model_name, 'Date': date_time, 'ACC': metrics.accuracy_score(y, pred)}
        eval_dict['FPR'], eval_dict['TPR'], thresholds = metrics.roc_curve(y, pred)
        eval_dict['AUC'] = metrics.auc(eval_dict['FPR'], eval_dict['TPR'])
        eval_dict['MCC'] = metrics.matthews_corrcoef(y, pred)
        eval_dict['F1_score'] = metrics.f1_score(y, pred)
        save_metrics(eval_dict)
        pred_res = pd.DataFrame(zip(pred, y), columns=['pred', 'y'])
        pred_res.to_csv(os.path.join(MODELS_PREDICTION_PATH, "pred_{0}_{1}.csv".format(model_name, date_time)),
                        index=False)