from notebooks.intra_transfer import run_intra_transfer
from notebooks.preprocessing import run_preprocessing
from notebooks.training import run_training
from notebooks.transferring import run_transfer
from src.visualization.graphs_handler import create_transfer_graphs

if __name__ == '__main__':
    # model_type = 'xgboost'

    for model_type in ['base','xgboost']:
        run_training(model_type)
        run_transfer(model_type = model_type)
        # run_intra_transfer(model_type = model_type)
    # create_transfer_graphs(compare_to_xgboost = True)
