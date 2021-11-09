from src.data.data_handler import *
from src.models.models_handler import *

def run_visualization():
    f_names = {'miRNA_Net': r"base_30_10_2021 16_00_37.csv", 'xgboost': r"xg_30_10_2021 18_29_07.csv"}
    # create_heatmaps(f_names)
    # calculate_std(f_names)
    create_transfer_graphs()



if __name__ == '__main__':
    run_visualization()
