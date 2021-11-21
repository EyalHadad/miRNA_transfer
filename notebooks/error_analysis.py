from src.analytics.analytics import model_diff,dataset_diff


def run_error_analysis():
    dataset_diff('cow1','human3')
    # dataset_diff('mouse2','worm1')
    # model_diff('mouse1', 'worm1')
    # model_diff('mouse2', 'cow1')





if __name__ == '__main__':
    run_error_analysis()