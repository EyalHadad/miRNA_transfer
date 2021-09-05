from src.models.models_handler import save_metrics,create_sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from constants import *
from time import gmtime, strftime
from src.models.model_learner import ModelLearner
from src.models.miRNA_transfer_subclass import miTransfer
import numpy as np
import pandas as pd
import logging
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
import tensorflow as tf
from tensorflow.python.keras.models import save_model
from tensorflow.python.keras.models import load_model
from src.models.models_handler import save_metrics,create_sequence,create_evaluation_dict




class Transfer_obj:
    feature_names=None
    src_model_name = None
    dst_org_name=None
    transfer_size = None
    l_model = None
    x = None
    y=None
    x_train=None
    y_train=None
    sequences=None
    x_val=None
    y_val=None
    sequences_tst=None

    def __init__(self, org_name):
        self.src_model_name = org_name
        self.l_model = miTransfer()


        print("set all layers to no trainable")
        for l in self.l_model.layers:
            print(l.name, l.trainable)
            if 'dense_8' not in l.name:
                l.trainable = False
        i=0


    def load_dst_data(self, dst_org_name):
        self.dst_org_name = dst_org_name
        train = pd.read_csv(os.path.join(PROCESSED_TRAIN_PATH, "{0}_train.csv".format(self.dst_org_name)), index_col=False)
        print("---Train data was loaded---\n")
        print("training data shape:", train.shape)
        train['sequence'] = train.apply(lambda x: create_sequence(x['miRNA sequence'], x['target sequence']), axis=1)
        self.y = train['label']
        X = train.drop(['mRNA_start', 'label', 'mRNA_name', 'target sequence', 'microRNA_name', 'miRNA sequence'],
                       axis=1)
        X.drop('sequence', 1).fillna(0, inplace=True)
        self.feature_names = list(X.columns)
        self.x = X


    def retrain_model(self, t_size):
        self.l_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        load_weights_path = os.path.join(MODELS_OBJECTS_PATH, f"{self.src_model_name}/")
        self.l_model.load_weights(load_weights_path)
        if t_size !=0:
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x, self.y, train_size=t_size, random_state=42)
            self.sequences = np.array(self.x_train['sequence'].values.tolist())
            self.x_train = self.x_train.drop('sequence', axis=1)
            self.sequences_tst = np.array(self.x_val['sequence'].values.tolist())
            self.x_val = self.x_val.drop('sequence', axis=1)
            self.x_train = self.x_train.astype("int")
            self.x_val = self.x_val.astype("float")
            self.l_model.fit([self.x_train, self.sequences], self.y_train, epochs=5)

        else:
            self.x_val, self.y_val = self.x, self.y
            self.sequences_tst = np.array(self.x_val['sequence'].values.tolist())
            self.x_val = self.x_val.drop('sequence', axis=1)
            self.x_val = self.x_val.astype("float")


        self.eval_model(t_size)



    def eval_model(self,t_size):
        print("Evaluate model")
        print(f" x shape:{self.x_val.shape} and seq sahpe: {self.sequences_tst.shape}")
        pred = self.l_model.predict([self.x_val, self.sequences_tst])
        date_time, model_name = create_evaluation_dict(self.src_model_name+"_"+str(t_size),self.dst_org_name,pred, self.y_val)
        pred_res = pd.DataFrame(zip(pred, self.y_val), columns=['pred', 'y'])
        pred_file_name = "pred_{0}_{1}_{2}.csv".format(self.src_model_name, t_size, self.dst_org_name)
        pred_res.to_csv(os.path.join(MODELS_PREDICTION_PATH, pred_file_name), index=False)




