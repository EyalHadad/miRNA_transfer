from notebooks.preprocessing import run_preprocessing
from notebooks.training import run_training
from notebooks.transferring import run_transfer
from src.visualization.graphs_handler import create_transfer_graphs

if __name__ == '__main__':
    # model_type = 'xgboost'

    model_type = 'base'
    run_preprocessing()
    in_res_dict = run_training(model_type)
    run_transfer(in_res_dict,model_type)
    create_transfer_graphs(compare_to_xgboost = True)
