from src.models.transfer.Base_transfer import BaseTransferObj
from src.models.transfer.xgboost_transfer import XgboostTransferObj
from src.models.models_handler import *


def run_transfer(in_res_dict=None, model_type='base'):
    dataset_list = copy.deepcopy(DATASETS)
    datasets_dict = {}
    transfer_size = TRANSFER_SIZE_LIST
    for org_name in dataset_list:
        rest = copy.deepcopy(DATASETS)
        rest.remove(org_name)
        if model_type == 'base':
            trans_obj = BaseTransferObj(org_name)
        else:
            trans_obj = XgboostTransferObj(org_name)
        datasets_dict[org_name] = {}
        for dst_org_name in rest:
            trans_obj.load_dst_data(dst_org_name)
            datasets_dict[org_name][dst_org_name] = {}
            for t_size in transfer_size:
                auc = trans_obj.retrain_model(t_size)
                datasets_dict[org_name][dst_org_name][t_size] = auc
    create_res_table(datasets_dict, in_res_dict)
    species_dict = create_species_dict(datasets_dict)
    for r in species_dict.keys():
        create_res_graph(species_dict[r], r, transfer_size,model_type)


if __name__ == '__main__':
    run_transfer()
i = 9
