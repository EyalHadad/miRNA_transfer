from src.models.models_handler import *
from src.visualization.graphs_handler import create_transfer_graphs, create_intra_transfer_graphs
from src.visualization.visualization_handler import create_heatmaps, save_statistic_file


def run_visualization():
    # cross_org_tables = {'miRNA_Net': r"base_11_11_2021 16_18_16.csv", 'xgboost': r"xgboost_11_11_2021 15_57_33.csv"}
    # TODO change loading tables
    cross_org_tables = {'miRNA_Net': r"base_19_11_2021 13_17_00.csv", 'xgboost': r"xgboost_19_11_2021 13_21_59.csv"}
    # create_heatmaps(cross_org_tables)
    model_list = ['base_20','xgboost_20','base_baseline','xgboost_baseline']
    # create_transfer_graphs(model_list)
    create_intra_transfer_graphs(model_list)




if __name__ == '__main__':
    run_visualization()
