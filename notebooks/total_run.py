from notebooks.preprocessing import run_preprocessing
from notebooks.training import run_training
from notebooks.transferring import run_transfer

if __name__ == '__main__':
    run_preprocessing()
    run_training()
    run_transfer()