from src.data.data_handler import *

if __name__ == '__main__':
    dataset_list = ['cow', 'human1', 'human2', 'human3', 'worm1', 'worm2', 'mouse1', 'mouse2']
    # dataset_list = ['worm1']
    for data in dataset_list:
        create_train_dataset(data)
