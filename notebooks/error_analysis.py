from src.analytics.analytics import model_diff,dataset_diff,create_shap_global_plots


def run_error_analysis():
    # create_shap_global_plots()
    dataset_diff()
    # dataset_diff('mouse2','worm1')
    # model_diff('mouse1', 'worm1')
    # model_diff('mouse2', 'cow1')





if __name__ == '__main__':
    run_error_analysis()