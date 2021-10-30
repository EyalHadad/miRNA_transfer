import pickle
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


class XgboostTransferObj(Transfer_obj):

    def __init__(self, org_name):
        Transfer_obj.__init__(self, org_name)
        model_name = os.path.join(MODELS_OBJECTS_PATH, '{0}_{1}.dat'.format(self.model_name, self.org_name))
        self.l_model = pickle.load(model_name)

    def retrain_model(self, t_size):
        super(XgboostTransferObj, self).retrain_model(t_size,'base')
