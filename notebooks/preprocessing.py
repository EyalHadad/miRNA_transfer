from src.data.data_handler import *


def run_preprocessing():
    dataset_list = DATASETS
    # dataset_list = ['worm1']
    for data in dataset_list:
        create_train_dataset(data, remove_hot_paring=False, only_most_important=True)


if __name__ == '__main__':
    run_preprocessing()
