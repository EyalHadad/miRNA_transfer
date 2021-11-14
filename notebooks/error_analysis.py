from src.analytics.analytics import create_error_datasets, analysis_error_datasets
from src.visualization.graphs_handler import create_intra_transfer_graphs


def run_error_analysis():
    # create_error_datasets(target_dataset='worm1')
    analysis_error_datasets(target_dataset='worm1')




if __name__ == '__main__':
    run_error_analysis()