import pandas as pd
import os
from constants import *
import numpy as np
import sklearn.model_selection as sk
from sklearn import preprocessing


def transform_categorical(data, categorical_features=None):
    for feat in categorical_features:
        le = preprocessing.LabelEncoder()
        le.fit(data[feat])
        data[feat] = le.transform(data[feat])
        return data


def get_data_from_file(data_file_name, is_pos):
    col_to_remove = ['Source', 'Organism', 'number of reads', 'mRNA_name', 'full_mrna', 'microRNA_name', 'miRNA sequence', 'target sequence']
    pos_or_neg = 'neg'
    if is_pos == 1:
        pos_or_neg = 'pos'
        col_to_remove.append('GI_ID')
    df = pd.read_csv(os.path.join(EXTERNAL_PATH, "{0}_{1}.csv".format(data_file_name, pos_or_neg)), index_col=0)
    print("{0} Original {1} data shape:{2}".format(pos_or_neg, data_file_name, df.shape))
    df.drop(col_to_remove, axis=1, inplace=True)
    df['label'] = is_pos
    print("Final {0} Data shape:{0}".format(pos_or_neg,df.shape))
    return df


def create_train_dataset(org_name):
    print("\n---Reading miRNA external data---")
    df_pos = get_data_from_file(data_file_name = org_name,is_pos=1)
    df_neg = get_data_from_file(data_file_name = org_name,is_pos=0)
    print("pos shape: {0} \n neg shape:{1}".format(df_pos.shape,df_neg.shape))
    df = pd.concat([df_pos,df_neg],join='inner' ,ignore_index=True)
    print("Total data shape:{0}".format(df.shape))
    df = df.sample(frac=1).reset_index(drop=True)
    train, test = sk.train_test_split(df, test_size=0.2)
    print("Training set shape is {0} and test is {1}\n".format(train.shape, test.shape))
    train.to_csv(os.path.join(PROCESSED_TRAIN_PATH, "{0}_train.csv".format(org_name)), index=False)
    print("---Train dataset was created---\n")
    test.to_csv(os.path.join(PROCESSED_TEST_PATH, "{0}_test.csv".format(org_name)), index=False)
    print("---Test dataset was created---\n")
    i=8
