from src.data.data_handler import *

if __name__ == '__main__':
    dataset_list = DATASETS
    # dataset_list = ['worm1']
    for data in dataset_list:
        create_train_dataset(data)
