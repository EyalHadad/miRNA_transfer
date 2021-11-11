import os
from statistics import mean

import pandas as pd

from constants import MODELS_OBJECTS_TRANSFER_TABLES
from src.models.models_handler import create_res_graph


def create_transfer_graphs(model_type):
    f_name = os.path.join(MODELS_OBJECTS_TRANSFER_TABLES, f"{model_type}_transfer.csv")
    data = pd.read_csv(f_name)
    transfer_avg_dict = data.to_dict()
    for s_org in transfer_avg_dict.keys():
        for d_org in transfer_avg_dict[s_org].keys():
            for s,val in transfer_avg_dict[s_org][d_org].items():
                org_list = transfer_avg_dict[s_org][d_org][s]
                transfer_avg_dict[s_org][d_org][s] = round(mean(org_list),2)

    for r in transfer_avg_dict.keys():
        create_res_graph(transfer_avg_dict[r], r, model_type,trans_epochs)

    return transfer_avg_dict