import xgboost as xgb
import os
from constants import *
import logging

logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
from src.models.transfer.Transfer_obj import Transfer_obj


class XgboostTransferObj(Transfer_obj):

    def __init__(self, org_name):
        Transfer_obj.__init__(self, org_name)
        self.l_model_path = os.path.join(MODELS_OBJECTS_PATH, 'Xgboost_{0}.dat'.format(org_name))
        self.l_model = xgb.XGBClassifier(kwargs=XGBS_PARAMS)  # init model

    def retrain_model(self, t_size,obj_type,trans_epochs):
        return super(XgboostTransferObj, self).retrain_model(t_size,'xgboost',trans_epochs)
