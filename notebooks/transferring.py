from src.models.transfer.Transfer_obj import Transfer_obj
from constants import *
import copy
from src.models.models_handler import create_res_table,create_res_graph
if __name__ == '__main__':

    dataset_list = copy.deepcopy(DATASETS)
    tabel_dict = {}
    transfer_size = TRANSFER_SIZE_LIST
    for org_name in dataset_list:
        rest = copy.deepcopy(DATASETS)
        rest.remove(org_name)
        trans_obj = Transfer_obj(org_name)
        tabel_dict[org_name] = {}
        for dst_org_name in rest:
            trans_obj.load_dst_data(dst_org_name)
            tabel_dict[org_name][dst_org_name] = {}
            for t_size in transfer_size:
                auc = trans_obj.retrain_model(t_size)
                tabel_dict[org_name][dst_org_name][t_size] = auc[1]
    create_res_table(tabel_dict)
    for r in tabel_dict.keys():
        create_res_graph(tabel_dict[r],r,dataset_list, transfer_size)
i=9
