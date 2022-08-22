"""Preprocessing dataset

This script allows the user to preprocess the selected dataset. 

It uses several preprocessing functions from other local modules with argparse input. 
Firstly it loads the dataset and clean some features separating the target one. Then, it splits
the dataset in train and test and save those instances of data as pickle objects in "data/interim" folder. 
In addition to doing that, remove outliers on train only and perform imputing, encoding and scaling
on train and test dataset without target.
Finally, it removes ignored features from train and test with and save those instances of data
as pickle objects in "data/processed" folder.

Usage
-----
    python preprocessing.py ../../data/raw/PAKDD2010_Modeling_Data.txt

Returns
-------
    Interim and processed instances of original dataset as pickle objects.
"""

# Python standard library
import argparse

# Third party libraries
import pandas as pd
import xlrd
import warnings

# Local libraries
from src.features import utils_features
from src import utils

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="load your dataset")
    parser.add_argument(
        "dataset_path",
        type=str,
        help=(
            "Full path to the file having all the applicants info. E.g. "
            "../data/raw/PAKDD2010_Modeling_Data.txt"
        ),
    )
    args = parser.parse_args()
    return args


def preprocess_data(dataset_path):
    # load and clean dataset from Nan, constant and empty features
    df = utils_features.load_dataset(dataset_path)
    df = utils_features.clean_dataset(df)

    # split dataset and get stratified target feature
    X_train, X_test, y_train, y_test = utils_features.split(
        df, "TARGET_LABEL_BAD=1"
    )

    # save dataset splits for data checkpoint
    utils.save_data_checkpoint(X_train, "../../data/interim/X_train.pickle")
    utils.save_data_checkpoint(y_train, "../../data/interim/y_train.pickle")
    utils.save_data_checkpoint(X_test, "../../data/interim/X_test.pickle")
    utils.save_data_checkpoint(y_test, "../../data/interim/y_test.pickle")

    # preprocess X_train and X_test features
    X_train = utils_features.outliers(X_train)
    X_train, X_test = utils_features.impute_nan(X_train, X_test)
    X_train, X_test = utils_features.scaling(X_train, X_test)
    X_train, X_test = utils_features.encode(X_train, X_test)

    # remove features with lot of classes to perform preprocessing
    ignored_features = [
        "CITY_OF_BIRTH",
        "RESIDENCIAL_CITY",
        "RESIDENCIAL_BOROUGH",
    ]
    X_train = X_train.drop(columns=ignored_features)
    X_test = X_test.drop(columns=ignored_features)

    # save dataset splits for data checkpoint
    utils.save_data_checkpoint(X_train, "../../data/processed/X_train.pickle")
    utils.save_data_checkpoint(X_test, "../../data/processed/X_test.pickle")
    utils.save_data_checkpoint(y_train, "../../data/processed/y_train.pickle")
    utils.save_data_checkpoint(y_test, "../../data/processed/y_test.pickle")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    args = parse_args()
    preprocess_data(args.dataset_path)
    print(
        "------------Dataset preprocessing complete successfully------------"
    )
