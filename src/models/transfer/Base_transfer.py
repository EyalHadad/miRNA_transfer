from src.models.models_handler import save_metrics, create_sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from constants import *
from time import gmtime, strftime
from src.models.model_learner import ModelLearner
from src.models.miRNA_transfer_subclass import miTransfer, api_model
import numpy as np
import pandas as pd
import logging
from keras.layers import Input
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
import tensorflow as tf
from tensorflow.python.keras.models import save_model
from tensorflow.python.keras.models import load_model
from src.models.models_handler import save_metrics, create_sequence, create_evaluation_dict
from keras.optimizers import SGD
from src.models.Transfer_obj import Transfer_obj

class BaseTransferObj(Transfer_obj):

    def __init__(self, org_name):
        self.l_model = api_model(MODEL_INPUT_SHAPE)
        print("set all layers to no trainable")
        for l in self.l_model.layers:
            print(l.name, l.trainable)
            l.trainable = False
        self.l_model.get_layer('dense_20').trainable=True
        self.l_model.get_layer('output').trainable=True


    def retrain_model(self, t_size):
        self.l_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        load_weights_path = os.path.join(MODELS_OBJECTS_PATH, f"{self.src_model_name}/")
        self.l_model.load_weights(load_weights_path)
        if t_size != 0:
            try:
                x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(self.x, self.y, train_size=t_size,
                                                                            random_state=42)
            except:
                return 0

            # TODO change second input to sequences - cuz they are not the same shape right now
            self.l_model.fit(x_train_t, y_train_t, epochs=10)

        auc = self.eval_model(t_size)
        return auc