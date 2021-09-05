from src.models.transfer.Transfer_obj import Transfer_obj
from constants import *


if __name__ == '__main__':
    # dataset_list = DATASETS
    dataset_list = ['worm1']
    rest = DATASETS
    rest.remove('worm1')
    rest = ['mouse1', 'mouse2']
    transfer_size = [0,100,200,500,800]
    for org_name in dataset_list:
        trans_obj = Transfer_obj(org_name)
        for dst_org_name in rest:
            trans_obj.load_dst_data(dst_org_name)
            for t_size in transfer_size:
                trans_obj.retrain_model(t_size)

