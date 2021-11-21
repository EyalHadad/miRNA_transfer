from src.models.models_handler import *
from src.data.data_handler import *
from scipy.stats import ks_2samp
import seaborn as sns
import matplotlib.pyplot as plt

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


def find_most_distinguish_cols(f_1,f_2,n=1):
    distinguish_list = []
    df1 = pd.read_csv(f_1).drop(FEATURES_TO_DROP, axis=1)
    df2 = pd.read_csv(f_2).drop(FEATURES_TO_DROP, axis=1)
    for c in df1:
        res = ks_2samp(df1[c], df2[c])
        distinguish_list.append([c,res.statistic,res.pvalue])
    distinguish_list.sort(key=lambda tup: tup[2])
    return [x[0] for x in distinguish_list[0:n]]


def plot_distinguish_cols(f1_path,f2_path,col_name,s_org,d_org):
    network_worng_pred = pd.read_csv(f1_path)
    xgboost_wrong_pred = pd.read_csv(f2_path)
    target_dataset = pd.read_csv(os.path.join(PROCESSED_TEST_PATH, "{0}_test.csv".format(d_org)), index_col=False)
    network_unique_misstake = network_worng_pred[~network_worng_pred.index.isin(xgboost_wrong_pred.index)]
    xgboost_unique_misstake = xgboost_wrong_pred[~xgboost_wrong_pred.index.isin(network_worng_pred.index)]
    network_worng_pred.loc[:,'dataset'] = 'Network'
    xgboost_wrong_pred.loc[:,'dataset'] = 'Xgboost'
    target_dataset.loc[:,'dataset'] = f"{s_org}_test_distribution"
    network_unique_misstake['dataset'] = 'Network unique mistakes'
    xgboost_unique_misstake['dataset'] = 'Xgboost unique mistakes'
    new_data = network_worng_pred.append(xgboost_wrong_pred).append(target_dataset).append(network_unique_misstake).append(xgboost_unique_misstake)
    sns.set_theme(style="darkgrid")
    sns.displot(data=new_data, x=col_name, hue='dataset',kind="kde")
    # plt.title(f"{s_org} model predict {d_org} interactions", fontsize=10)
    plt.savefig(os.path.join(MODELS_ERROR_STATISTICS, f"{d_org}_{col_name}_wrong_prediction.png"))
    plt.clf()


def model_diff(s_org,d_org):
    dict_names = {'base': f"base_{s_org}_{d_org}_incorrect.csv",
                  'xgboost': f"xgboost_{s_org}_{d_org}_incorrect.csv"}

    nn_path = os.path.join(MODELS_PREDICTION_PATH, dict_names['base'])
    xg_path = os.path.join(MODELS_PREDICTION_PATH, dict_names['xgboost'])
    col_names = find_most_distinguish_cols(nn_path, xg_path, n=10)
    for c in col_names:
        plot_distinguish_cols(nn_path,xg_path, c, s_org, d_org)


def dataset_diff(s_org,d_org ):
    s_file_path = os.path.join(PROCESSED_TEST_PATH, "{0}_test.csv".format(d_org))
    d_data = pd.read_csv(s_file_path, index_col=False)
    d_data = d_data.drop(FEATURES_TO_DROP, axis=1)
    d_file_path = os.path.join(PROCESSED_TRAIN_PATH, "{0}_train.csv".format(s_org))
    s_data = pd.read_csv(d_file_path, index_col=False)
    s_data = s_data.drop(FEATURES_TO_DROP, axis=1)
    # col_names = find_most_distinguish_cols(s_file_path, d_file_path)
    col_names = ['Energy_MEF_local_target']
    d_data.loc[:, 'dataset'] = f"{d_org}_test_distribution"
    s_data.loc[:, 'dataset'] = f"{s_org}_train_distribution"
    new_data = d_data.append(s_data)
    for c in col_names:
        sns.set_theme(style="darkgrid")
        sns.displot(data=new_data, x=c, hue='dataset', kind="kde")
        plt.savefig(os.path.join(MODELS_ERROR_STATISTICS, f"{s_org}_{d_org}_dataset_diff.png"))
        plt.clf()
