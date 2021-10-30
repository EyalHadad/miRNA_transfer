from notebooks.preprocessing import run_preprocessing
from notebooks.training import run_training
from notebooks.transferring import run_transfer

if __name__ == '__main__':
    model_type = 'xgboost'
    # run_preprocessing()
    in_res_dict = run_training(model_type)
    run_transfer(in_res_dict,model_type)