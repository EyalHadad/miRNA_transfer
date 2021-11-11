from src.models.transfer.Base_transfer import BaseTransferObj
from src.models.transfer.xgboost_transfer import XgboostTransferObj
from src.models.models_handler import *


def run_transfer(model_type='base', trans_epochs=20):
    dataset_list = copy.deepcopy(DATASETS)
    transfer_dict = {}
    vanilla_model_dict = {}
    transfer_size = TRANSFER_SIZE_LIST
    for org_name in dataset_list:
        rest = copy.deepcopy(DATASETS)
        rest.remove(org_name)
        if model_type == 'base':
            trans_obj = BaseTransferObj(org_name)
        else:
            trans_obj = XgboostTransferObj(org_name)
        transfer_dict[org_name] = {}
        vanilla_model_dict[org_name] = {}
        for dst_org_name in rest:
            trans_obj.load_dst_data(dst_org_name)
            transfer_dict[org_name][dst_org_name] = {}
            vanilla_model_dict[org_name][dst_org_name] = {}
            for t_size in transfer_size:
                v_auc = trans_obj.train_new_model(t_size, model_type)
                vanilla_model_dict[org_name][dst_org_name][t_size] = v_auc
                auc = trans_obj.retrain_model(t_size,model_type,trans_epochs)
                transfer_dict[org_name][dst_org_name][t_size] = auc

    save_cross_org_table(transfer_dict, model_type)
    save_transfer_table(transfer_dict, model_type, trans_epochs)
    save_transfer_table(vanilla_model_dict, model_type, 'baseline')



if __name__ == '__main__':
    run_transfer(model_type='xgboost', trans_epochs=20)
    run_transfer(model_type='base', trans_epochs=20)
    # for i in range(20,120,20):
        # TRANS_EPOCHS = i
        # run_transfer(model_type='base', trans_epochs=i)
        # print(i)
i = 9
