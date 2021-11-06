import matplotlib.pyplot as plt
import os
from keras.optimizers import SGD
from constants import *
from time import gmtime, strftime
from src.models.model_learner import ModelLearner
from src.models.miRNA_transfer_subclass import miTransfer,api_model
import numpy as np
import pandas as pd
import logging
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
import tensorflow as tf
from tensorflow.python.keras.models import save_model,load_model
from keras.layers import Input

class BaseTrainObj(ModelLearner):
    history = None

    def __init__(self, org_name, m_name):
        ModelLearner.__init__(self, org_name, m_name)

    def train_model(self):
        super().prep_model_training()
        print("---Start training {0} on {1}---\n".format(self.model_name, self.org_name))
        print(f"ann input {self.x.shape}, cnn input {self.sequences.shape}")
        self.model = api_model(self.x.shape[-1],)
        self.model.compile(optimizer=SGD(lr=0.01, momentum=0.9, clipnorm=1.0), loss='binary_crossentropy',metrics=['acc'])
        self.history = self.model.fit(self.x, self.y, epochs=TRAIN_EPOCHS,validation_data=(self.xval, self.yval))
        # print(self.model.summary())
        print("---Learning Curves---\n")
        self.plot_learning_curves()
        model_name = os.path.join(MODELS_OBJECTS_PATH,f"{self.org_name}/")
        print("---Saving model---\n")
        self.model.save_weights(model_name)
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
        super().model_explain()

