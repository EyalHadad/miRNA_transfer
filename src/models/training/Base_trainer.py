import matplotlib.pyplot as plt
import os
from constants import *
from time import gmtime, strftime
from src.models.model_learner import ModelLearner
from src.models.vanilla_models_configurations import base_learning,cnn_learning2,rnn_learning, miTrans_base_model
import numpy as np
import pandas as pd
import logging
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

from tensorflow.python.keras.models import save_model


class BaseTrainObj(ModelLearner):
    history = None

    def __init__(self, org_name, m_name):
        ModelLearner.__init__(self, org_name, m_name)

    def train_model(self):
        super().prep_model_training()
        print("---Start training {0} on {1}---\n".format(self.model_name, self.org_name))

        self.model = miTrans_base_model(self.x.shape[1])
        self.history = self.model.fit([self.x, self.sequences], self.y, epochs=5,
                                      validation_data=([self.xval, self.sequences_tst], self.yval))

        # self.model = cnn_learning2()
        # self.model = rnn_learning()
        # self.history = self.model.fit(self.sequences, self.y, batch_size=256, epochs=5, verbose=2, validation_data=(self.sequences_tst, self.yval))

        print("---Learning Curves---\n")
        self.plot_learning_curves()
        model_name = os.path.join(MODELS_OBJECTS_PATH,
                                  '{0}_{1}_{2}'.format(self.model_name, self.org_name, strftime("%Y-%m-%d", gmtime())))
        save_model(self.model, model_name)
        print("---{0} model saved---\n".format(self.model_name))

    def plot_learning_curves(self):
        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(1, len(acc) + 1)
        plt.figure()
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Accuracy ' + self.model_name)
        plt.legend()  # for the two lines, can get parameters such as loc='upper left' for locating the lines "menu"
        plt.savefig(os.path.join(MODELS_OUTPUT_PATH, 'Base {0} Acc.png'.format(self.org_name)))
        plt.figure()
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'bo', label='Validation loss')
        plt.title('Loss ' + self.model_name)
        plt.legend()
        plt.savefig(os.path.join(MODELS_OUTPUT_PATH, 'Base {0} Loss.png'.format(self.org_name)))
        plt.clf()

    def model_explain(self):
        print("---Explain model---\n")
        self.feature_importance()
        super().model_explain()

    def feature_importance(self):
        print("feature_importances\n")
        importance = self.model.feature_importances_
        f_important = sorted(list(zip(self.feature_names, importance)), key=lambda x: x[1], reverse=True)
        plt.bar([x[0] for x in f_important[:5]], [x[1] for x in f_important[:5]])
        plt.xticks(rotation=20)
        title = '{0} {1} f_important'.format(self.model_name, self.org_name)
        plt.title(title)
        plt.savefig(os.path.join(MODELS_FEATURE_IMPORTANCE, '{0}.png'.format(title)))
        plt.clf()
