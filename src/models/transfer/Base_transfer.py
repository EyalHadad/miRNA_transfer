from src.models.models_handler import create_sequence
from src.models.csv_handler import save_metrics
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
        Transfer_obj.__init__(self, org_name)
        self.l_model = api_model(MODEL_INPUT_SHAPE)
        print("set all layers to no trainable")
        for l in self.l_model.layers:
            print(l.name, l.trainable)
            l.trainable = False
        self.l_model.get_layer('dense_20').trainable = True
        self.l_model.get_layer('output').trainable = True

    def retrain_model(self, t_size, obj_type, trans_epochs):
        self.l_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        load_weights_path = os.path.join(MODELS_OBJECTS_PATH, f"{self.src_model_name}/")
        self.l_model.load_weights(load_weights_path)
        return super(BaseTransferObj, self).retrain_model(t_size, 'base', trans_epochs)
