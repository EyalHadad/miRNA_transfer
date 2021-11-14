from src.models.models_handler import *
from src.data.data_handler import *


def create_error_file(org_name, target_dataset, x, y):
    for model_type in ['base', 'xgboost']:
        model = load_trained_model(model_type, org_name)
        print(f"Predicting {target_dataset} using {model_type}_{org_name}")
        pred = model.predict(x)
        total_frame = x.copy()
        total_frame["index"] = x.index
        total_frame["actual"] = y
        total_frame["predicted"] = np.round(pred)
        incorrect = total_frame[total_frame["actual"] != total_frame["predicted"]]
        incorrect.to_csv(
            os.path.join(MODELS_WRONG_PREDICTIONS, f"{target_dataset}_{org_name}_{model_type}_incorrect.csv"),
            index=False)


def get_target_test_data(target_dataset):
    test = pd.read_csv(os.path.join(PROCESSED_TEST_PATH, "{0}_test.csv".format(target_dataset)), index_col=False)
    test['sequence'] = test.apply(lambda x: create_sequence(x['miRNA sequence'], x['target sequence']), axis=1)
    y = test['label']
    X = test.drop(FEATURES_TO_DROP, axis=1)
    X.drop('sequence', axis=1, inplace=True)
    x = X.astype("float")
    return x, y


def create_error_datasets(target_dataset):
    x_test, y_test = get_target_test_data(target_dataset)
    for org_name in DATASETS:
        print(f"Analysis {org_name}")
        create_error_file(org_name, target_dataset, x_test, y_test)


def analysis_error_datasets(target_dataset):
    csv_files = [x for x in os.listdir(MODELS_WRONG_PREDICTIONS) if f"{target_dataset}_" in x]
    wrong_pred_dict = create_wrong_dict(csv_files)
    write_statistics_file(target_dataset, wrong_pred_dict)


def write_statistics_file(target_dataset, wrong_pred_dict):
    f_name = os.path.join(MODELS_ERROR_STATISTICS, f"{target_dataset}_statistics.csv")
    with open(f_name, 'w') as file:
        file.write("s_org,stat,miRNA_Net,Xgboost\n")
        for t_org, v in wrong_pred_dict.items():
            base_wrong = wrong_pred_dict[t_org]['base']
            xg_wrong = wrong_pred_dict[t_org]['xgboost']
            amount_row = f"{t_org},#mis predictions,{base_wrong.shape[0]},{xg_wrong.shape[0]}\n"
            file.write(amount_row)
            fp_row = f"{t_org},#false positive,{base_wrong[base_wrong['predicted'] == 1].shape[0]},{xg_wrong[xg_wrong['predicted'] == 1].shape[0]}\n"
            file.write(fp_row)
            u_miss_base = base_wrong[~base_wrong.index.isin(xg_wrong.index)].shape[0]
            u_miss_xg = xg_wrong[~xg_wrong.index.isin(base_wrong.index)].shape[0]
            unique_rows = f"{t_org},#unique mistakes,{u_miss_base},{u_miss_xg}\n"
            file.write(unique_rows)


def create_wrong_dict(csv_files):
    wrong_pred_dict = {}
    for file in csv_files:
        data = pd.read_csv(os.path.join(MODELS_WRONG_PREDICTIONS, file), index_col=['index'])
        t_org = file.split('_')[1]
        if t_org not in wrong_pred_dict.keys():
            wrong_pred_dict[t_org] = {}
        model_type = file.split('_')[2]
        wrong_pred_dict[t_org][model_type] = data
    return wrong_pred_dict

