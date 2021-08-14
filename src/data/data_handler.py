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


def create_train_dataset():
    print("\n---Reading miRNA external data---")
    df = pd.read_csv(os.path.join(EXTERNAL_PATH, "data3.csv"))
    print("Original data shape:{0}".format(df.shape))
    df = transform_categorical(data=df, categorical_features=['region', 'paper region'])
    print("transformed data shape:{0}".format(df.shape))
    df = df.select_dtypes(exclude=['object'])
    print("Object excluded data shape:{0}".format(df.shape))
    df.drop(['key', 'index','miRNA ID'], axis = 1,inplace=True)
    print("key and index excluded data shape:{0}".format(df.shape))
    df['label'] = np.random.randint(0, 2, df.shape[0])
    print("Final data shape:{0}".format(df.shape))
    print("There are {0} positives and {1} negatives\n".format(len(df[df['label'] == 1]), len(df[df['label'] == 0])))
    train, test = sk.train_test_split(df, test_size=0.2)
    print("Training set shape is {0} and test is {1}\n".format(train.shape, test.shape))
    train.to_csv(os.path.join(PROCESSED_TRAIN_PATH, "train.csv"), index=False)
    print("---Train dataset was created---\n")
    test.to_csv(os.path.join(PROCESSED_TEST_PATH, "test.csv"), index=False)
    print("---Test dataset was created---\n")
