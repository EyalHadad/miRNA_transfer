from src.models.transfer.Transfer_obj import Transfer_obj
from src.models.models_handler import *


def run_transfer():
    dataset_list = copy.deepcopy(DATASETS)
    datasets_dict = {}
    transfer_size = TRANSFER_SIZE_LIST
    for org_name in dataset_list:
        rest = copy.deepcopy(DATASETS)
        rest.remove(org_name)
        trans_obj = Transfer_obj(org_name)
        datasets_dict[org_name] = {}
        for dst_org_name in rest:
            trans_obj.load_dst_data(dst_org_name)
            datasets_dict[org_name][dst_org_name] = {}
            for t_size in transfer_size:
                auc = trans_obj.retrain_model(t_size)
                datasets_dict[org_name][dst_org_name][t_size] = auc[1]
    create_res_table(datasets_dict)
    species_dict = create_species_dict(datasets_dict)
    for r in species_dict.keys():
        create_res_graph(species_dict[r], r, transfer_size)


if __name__ == '__main__':
    run_transfer()
i=9
