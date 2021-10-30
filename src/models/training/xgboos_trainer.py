import matplotlib.pyplot as plt
import xgboost as xgb
import os
from constants import *
from time import gmtime, strftime
from src.models.model_learner import ModelLearner
from src.models.models_handler import save_feature_importance_res


class XgboostTrainObj(ModelLearner):

    def __init__(self,org_name,m_name):
        ModelLearner.__init__(self,org_name,m_name)


    def train_model(self):
        super().prep_model_training()
        print("---Start training {0} on {1}---\n".format(self.model_name,self.org_name))
        self.model = xgb.XGBClassifier().fit(self.x, self.y, eval_metric=["error", "logloss"], eval_set=[(self.x, self.y), (self.xval, self.yval)])
        print("---Learning Curves---\n")
        self.plot_learning_curves()
        model_name = os.path.join(MODELS_OBJECTS_PATH,'{0}_{1}.json'.format(self.model_name,self.org_name))
        self.model.save_model(model_name)
        print("---{0} model saved---\n".format(self.model_name))


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
        print("---Explain model---\n")
        self.feature_importance()
        super().model_explain()


    def feature_importance(self):
        print("feature_importances\n")
        importance = self.model.feature_importances_
        f_important = sorted(list(zip(self.feature_names, importance)), key=lambda x: x[1], reverse=True)
        save_feature_importance_res('{0}_{1}'.format(self.model_name,self.org_name),f_important,'reg')

        plt.bar([x[0] for x in f_important[:5]], [x[1] for x in f_important[:5]])
        plt.xticks(rotation=20)
        title = '{0} {1} f_important'.format(self.model_name, self.org_name)
        plt.title(title)
        plt.savefig(os.path.join(MODELS_FEATURE_IMPORTANCE, '{0}.png'.format(title)))
        plt.clf()

