from sklearn.inspection import permutation_importance
import scikitplot as skplt
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import shap
import numpy as np
from constants import *
from sklearn.model_selection import train_test_split
import os
from src.models.models_handler import create_sequence, create_evaluation_dict, load_trained_model
from src.models.csv_handler import save_metrics


class ModelLearner:
    x = None
    mRNA = None
    miRNA = None
    xval = None
    y = None
    yval = None
    model = None
    org_name = None
    feature_names = None
    model_name = None
    sequences = None
    sequences_tst = None

    def __init__(self, org_name, m_name):
        self.org_name = org_name
        self.model_name = m_name

    def prep_model_training(self, aa1=False):
        if aa1:
            train = pd.read_csv(os.path.join(PROCESSED_ALL_AGAINST_PATH, "{0}_train.csv".format(self.org_name)), index_col=False)
        else:
            train = pd.read_csv(os.path.join(PROCESSED_TRAIN_PATH, "{0}_train.csv".format(self.org_name)), index_col=False)
        # train = train.iloc[:500, :]
        print("---Train data was loaded---\n")
        print("training data shape:", train.shape)
        train['sequence'] = train.apply(lambda x: create_sequence(x['miRNA sequence'], x['target sequence']), axis=1)
        y = train['label']
        X = train.drop(FEATURES_TO_DROP, axis=1)
        X.drop('sequence', 1).fillna(0, inplace=True)

        self.feature_names = list(X.columns)
        self.x, self.xval, self.y, self.yval = train_test_split(X, y, test_size=0.2, random_state=42)

        self.sequences = np.array(self.x['sequence'].values.tolist())
        self.x = self.x.drop('sequence', axis=1)
        self.sequences_tst = np.array(self.xval['sequence'].values.tolist())
        self.xval = self.xval.drop('sequence', axis=1)
        self.x.fillna(0, inplace=True)
        self.x = self.x.astype("float")
        self.xval.fillna(0, inplace=True)
        self.xval = self.xval.astype("float")

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
        explainer = shap.DeepExplainer(self.model, self.x.values.astype('float'))
        shap_values = explainer.shap_values(self.xval.iloc[1:5, ].values)
        shap.summary_plot(shap_values, feature_names=self.xval.iloc[1:5, ].columns, plot_type="bar", show=False,
                          max_display=10, plot_size=(20, 20))
        plt.title('{0} {1} SHAP bar'.format(self.model_name, self.org_name))
        plt.savefig(os.path.join(MODELS_FEATURE_IMPORTANCE, '{0}_{1}_bar.png'.format(self.model_name, self.org_name)))
        plt.clf()
        print("---Shap plots were saved---\n")

    def calc_permutation_importance(self):
        imps = permutation_importance(self.model, self.xval, self.yval)
        importances = imps.importances_mean
        std = imps.importances_std
        indices = np.argsort(importances)[::-1]
        title = '{0} {1} f_important'.format(self.model_name, self.org_name)
        # TODO move it to handler file (plot_figure)- need to know first what argument to send
        plt.figure(figsize=(10, 7))
        plt.title(title)
        plt.bar(range(self.xval.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
        # plt.xticks(range(X_test.shape[1]), [features[indices[i]] for i in range(6)])
        # plt.xlim([-1, X_test.shape[1]])
        plt.savefig(os.path.join(MODELS_FEATURE_IMPORTANCE, '{0}_bar.png'.format(title)))
        plt.clf()

    def evaluate_model(self, model_type,aa1=False):
        if aa1:
            test = pd.read_csv(os.path.join(PROCESSED_ALL_AGAINST_PATH, "{0}_test.csv".format(self.org_name)), index_col=False)
        else:
            test = pd.read_csv(os.path.join(PROCESSED_TEST_PATH, "{0}_test.csv".format(self.org_name)), index_col=False)
        print("---Test data was loaded---\n")
        test['sequence'] = test.apply(lambda x: create_sequence(x['miRNA sequence'], x['target sequence']), axis=1)
        y = test['label']
        X = test.drop(FEATURES_TO_DROP, axis=1)
        X.drop('sequence', axis=1, inplace=True)
        x = X.astype("float")
        if self.model is None:
            self.model = load_trained_model(model_type, self.org_name)

        if model_type == 'base':
            pred = self.model.predict(x)
        else:
            pred = self.model.predict_proba(x)[:,1]
        date_time, model_name, auc = create_evaluation_dict(self.model_name, self.org_name, pred, y)
        # total_frame = x.copy()
        # total_frame["index"] = x.index
        # total_frame["actual"] = y
        # total_frame["predicted"] = np.round(pred)
        # incorrect = total_frame[total_frame["actual"] != total_frame["predicted"]]
        # incorrect.to_csv(
        #     os.path.join(MODELS_PREDICTION_PATH, f"{model_type}_{self.org_name}_crossvalid_incorrect.csv"),
        #     index=False)
        # self.plot_roc_curve(model_type, pred, y)

        # pred_res = pd.DataFrame(zip(pred, y), columns=['pred', 'y'])
        # pred_res.to_csv(os.path.join(MODELS_PREDICTION_PATH, "pred_{0}_{1}.csv".format(model_name, date_time)),
        #                 index=False)
        return model_name, auc

    def plot_roc_curve(self, model_type, pred, y):
        from sklearn.metrics import roc_curve
        fpr1, tpr1, thresh1 = roc_curve(y, pred, pos_label=1)
        random_probs = [0 for i in range(len(y))]
        p_fpr, p_tpr, _ = roc_curve(y, random_probs, pos_label=1)
        plt.style.use('seaborn')
        plt.plot(fpr1, tpr1, linestyle='--', color='orange', label='Model')
        plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        # skplt.metrics.plot_roc_curve(y.to_numpy().reshape(y.shape[0],1), pred)
        plt.savefig(os.path.join(MODELS_OUTPUT_PATH, f"ROC_{model_type}_{self.org_name}.png"))
        plt.clf()
