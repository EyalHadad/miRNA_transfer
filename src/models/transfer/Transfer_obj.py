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


class Transfer_obj:
    feature_names = None
    src_model_name = None
    dst_org_name = None
    transfer_size = None
    l_model = None
    x = None
    y = None
    x_train = None
    y_train = None
    sequences = None
    x_test = None
    y_test = None
    sequences_tst = None

    def __init__(self, org_name):
        self.src_model_name = org_name
        self.l_model = api_model()

        print("set all layers to no trainable")
        for l in self.l_model.layers:
            print(l.name, l.trainable)
            l.trainable = False
        self.l_model.get_layer('dense_20').trainable=True
        self.l_model.get_layer('output').trainable=True


    def load_dst_data(self, dst_org_name):
        self.dst_org_name = dst_org_name
        train = pd.read_csv(os.path.join(PROCESSED_TRAIN_PATH, "{0}_train.csv".format(self.dst_org_name)),
                            index_col=False)
        test = pd.read_csv(os.path.join(PROCESSED_TEST_PATH, "{0}_test.csv".format(self.dst_org_name)), index_col=False)
        print("---Data was loaded---\n")
        print("Train data shape:", train.shape)
        print("Test data shape:", test.shape)
        train['sequence'] = train.apply(lambda x: create_sequence(x['miRNA sequence'], x['target sequence']), axis=1)
        test['sequence'] = test.apply(lambda x: create_sequence(x['miRNA sequence'], x['target sequence']), axis=1)
        self.y = train['label']
        self.y_test = test['label']
        self.sequences = np.array(train['sequence'].values.tolist())
        self.sequences_tst = np.array(test['sequence'].values.tolist())
        X = train.drop(
            ['mRNA_start', 'label', 'mRNA_name', 'target sequence', 'microRNA_name', 'miRNA sequence', 'full_mrna'],
            axis=1)
        self.feature_names = list(X.columns)
        self.x = X.drop('sequence', 1).fillna(0)
        self.x = self.x.astype("float")
        X_test = test.drop(
            ['mRNA_start', 'label', 'mRNA_name', 'target sequence', 'microRNA_name', 'miRNA sequence', 'full_mrna'],
            axis=1)
        self.x_test = X_test.drop('sequence', 1).fillna(0)
        self.x_test = self.x_test.astype("float")

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
            self.l_model.fit([x_train_t, x_train_t], y_train_t, epochs=10)

        auc = self.eval_model(t_size)
        return auc

    def eval_model(self, t_size):
        print("Evaluate model")
        # TODO change second input to sequences - cuz they are not the same shape right now
        # print(f" x shape:{self.x_test.shape} and seq sahpe: {self.sequences_tst.shape}")
        pred = self.l_model.predict([self.x_test, self.x_test])
        date_time, org_name, auc = create_evaluation_dict(self.src_model_name + "_" + str(t_size), self.dst_org_name,
                                                          pred, self.y_test)
        pred_res = pd.DataFrame(zip(pred, self.y_test), columns=['pred', 'y'])
        pred_file_name = "pred_{0}_{1}_{2}.csv".format(self.src_model_name, t_size, self.dst_org_name)
        pred_res.to_csv(os.path.join(MODELS_PREDICTION_PATH, pred_file_name), index=False)
        return auc
