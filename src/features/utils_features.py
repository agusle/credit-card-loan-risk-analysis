# Third Party libraries
import pandas as pd
import xlrd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Local libraries
from src.utils import save_data_checkpoint


def load_dataset(dataset):
    """
    Load dataset as pandas dataframe with features names of
    VariablesList.XLS file's features as columns. Also, check for duplicate
    features and replace duplicated ones adding "_2" to it's name. In addition,
    "ID_CLIENT" feature is assigned as the dataframe's index.

    Parameters
    ----------
    dataset: .txt / .csv
      Dataset file with tabular text (separated by tabs).

    Returns
    -------
    dataset: pandas.DataFrame
      Pandas dataframe with 53 total features and ID as dataframe's index.
    """

    # take headers from VariablesList.XLS
    headers_df = pd.read_excel("../../data/raw/PAKDD2010_VariablesList.XLS")
    headers = headers_df["Var_Title"].tolist()

    # replace duplicate headers
    headers_unique = []
    for header in headers:
        if header not in headers_unique:
            headers_unique.append(header)
        else:
            header_duplicated = header + "_2"
            headers_unique.append(header_duplicated)

    # create train dataframe with headers
    dataset = pd.read_csv(
        dataset,
        sep="\t",
        encoding="unicode_escape",
        low_memory=False,
        index_col="ID_CLIENT",
        names=headers_unique,
    )
    return dataset


def clean_dataset(df):
    """
    Clean a dataset as pd.DataFrame from Constant, NaN and empty features.
    Firstly, it identifies the categorical, categorical numerical and numerical features
    that match any of the status mentioned. Finally, drop all identified features from dataframe.

    Parameters
    ----------
    df: pandas.DataFrame
      Raw dataset

    Returns
    -------
    df: pandas.DataFrame
      Cleaned dataset

    """
    cat_features = df.select_dtypes("O").nunique()
    cat_num_features = df.select_dtypes("number").nunique()[
        df.select_dtypes("number").nunique() < 15
    ]

    # create lists with  constant classes, features with +50% empty or NaN values
    cat_constant = list(cat_features[cat_features == 1].index)
    num_constant = ["MONTHS_IN_THE_JOB"]
    catnum_constant = list(cat_num_features[cat_num_features == 1].index)
    catnum_constant_dist = [
        "POSTAL_ADDRESS_TYPE",
        "FLAG_DINERS",
        "FLAG_AMERICAN_EXPRESS",
        "FLAG_OTHER_CARDS",
    ]
    empty_features = list(df.eq(" ").sum()[df.eq(" ").sum() > 25000].index)
    nan_features = list(df.isna().sum()[df.isna().sum() > 25000].index)

    # grouping lists in only one
    remove_features = [
        *nan_features,
        *empty_features,
        *catnum_constant,
        *catnum_constant_dist,
        *num_constant,
        *cat_constant,
    ]
    # remove selected features from dataframes
    df = df.drop(remove_features, axis=1)
    return df


def split(df, label):
    """
    Split original dataset into train and test in 80/20 rate and separating both by target values.
    The split is stratified on target value to mantain proportions of the same in the returned datasets.

    Parameters
    ----------
    df: pandas.DataFrame
      Cleaned dataset (without NaN,empty and constant features).

    label: str
      Name of target value feature.

    Returns
    -------
    X_train: pandas.DataFrame
      80% train features values

    X_test: pandas.DataFrame
      20% test features values

    y_train: pandas.DataFrame
      80% train target values

    y_test: pandas.DataFrame
      20% test target values
    """

    X_train, X_test, y_train, y_test = train_test_split(
        df, df[label], test_size=0.20, random_state=42, stratify=df[label]
    )
    X_train = X_train.drop(label, axis=1)
    X_test = X_test.drop(label, axis=1)

    return X_train, X_test, y_train, y_test


def outliers(df):
    """
    Replace outliers of following skewed distribution features
    by Numpy NaN values:

    ['OTHER_INCOMES','QUANT_DEPENDANTS', 'AGE']

    Fit features classes with delimited values on PAKDD2010_VariablesList.XLS
    replacing the incorrect ones with numpy.NaN.

    Parameters
    ----------
    df: pandas.DataFrame
      Train split from dataset of features values without target.

    Returns
    -------
    df: pandas.DataFrame
      Train split from dataset with outliers converted to numpy.NaN.
    """

    df["OTHER_INCOMES"][df["OTHER_INCOMES"] > 190000.0] = np.NaN
    df["QUANT_DEPENDANTS"][df["QUANT_DEPENDANTS"] > 20] = np.NaN
    df["AGE"][df["AGE"] < 17] = np.NaN

    # adapt feature classes to PAKDD2010_VariablesList.XLS
    df["SEX"][df["SEX"] == "N"] = np.NaN
    df["STATE_OF_BIRTH"][df["STATE_OF_BIRTH"] == "XX"] = np.NaN
    df["MARITAL_STATUS"][df["MARITAL_STATUS"] == 0] = np.NaN
    df["RESIDENCE_TYPE"][df["RESIDENCE_TYPE"] == 0.0] = np.NaN
    df["OCCUPATION_TYPE"][df["OCCUPATION_TYPE"] == 0.0] = np.NaN

    return df


def impute_nan(train, test):
    """
    This function replace NaN values depending on feature dtype:

    Numerical feature (number): replace with feature's median.
    Categorical feature (object): replace with feature's mode.

    Parameters
    ----------
    train: pandas.DataFrame
      Train split from dataset with outliers replaced with NaN.

    test: pandas.DataFrame
      Test split from dataset with outliers replaced with NaN.

    Returns
    -------
    train: pandas.DataFrame
      Train split with NaN values replaced.

    test: pandas.DataFrame
      Test split with NaN values replaced.
    """

    # Replace empty string values with np.NAN.
    train = train.replace(" ", np.NaN)
    test = test.replace(" ", np.NaN)

    # create a dataframe with sum of NaN values of numerical features
    missing_num_col = pd.DataFrame(
        train.select_dtypes(include="number").isna().sum(), columns=["NaN"]
    )
    # filter dataframe by feature's name with Nan values != 0
    missing_num_col = missing_num_col["NaN"][
        (missing_num_col["NaN"] != 0)
    ].index

    # input median values for all numerical columns with missing data
    for feature in missing_num_col:
        train[feature] = train[feature].fillna(train[feature].median())
        test[feature] = test[feature].fillna(test[feature].median())

    # create a dataframe with sum of NaN values of non numerical features
    missing_non_numerical = pd.DataFrame(
        train.select_dtypes(include="O").isna().sum(), columns=["NaN"]
    )
    # filter dataframe by feature's name with Nan values != 0
    missing_non_numerical = missing_non_numerical["NaN"][
        (missing_non_numerical["NaN"] != 0)
    ].index

    # input mode values for all non numerical columns with missing data
    for feature in missing_non_numerical:
        train[feature] = train[feature].fillna(train[feature].mode().loc[0])
        test[feature] = test[feature].fillna(test[feature].mode().loc[0])

    return train, test


def scaling(train, test):
    """
    Scale numerical features using Sklearn StandardScaler.
    Firstly, create a list of numerical features without categorical numerical ones.
    Finally, scale all numerical features and save scaler as pickle object
    in "../models/utils/" for future use.

    Parameters
    ----------
    train: pandas.DataFrame
      Train split from dataset without NaN values.

    test: pandas.DataFrame
      Test split from dataset without NaN values.

    Returns
    -------
    train: pandas.DataFrame
      Train split from dataset with scaled numerical features.

    test: pandas.DataFrame
      Test split from dataset with scaled numerical features.
    """

    # create a list of features to scale without categorical numerical features
    cat_num_features = train.select_dtypes("number").nunique()[
        train.select_dtypes("number").nunique() < 15
    ]
    number_dtype = train.select_dtypes("number").columns
    num_features_list = list(set(number_dtype) - set(cat_num_features.index))
    num_features_list.remove("PROFESSION_CODE")

    # instance scaler
    scaler = StandardScaler()

    # scale column by column with each corresponding fit between train and test
    for feature in num_features_list:
        train[feature] = scaler.fit_transform(train[[feature]])
        test[feature] = scaler.transform(test[[feature]])

    # save fitted scaler to preprocess new data
    save_data_checkpoint(scaler, "../models/utils/scaler.pickle")

    return train, test


def encode(train, test):
    """
    Encode categorical features using sklearn One Hot Encoder.
    It ignore following features with more than 1k classes dropping those from datasets:
    ['CITY_OF_BIRTH','RESIDENCIAL_CITY','RESIDENCIAL_BOROUGH']
    The encoder is used dropping the first column.
    Finally, the encoded features are dropped from datasets and the encoder is saved
    as a pickle object in "../models/utils" for future use.

    Parameters
    ----------
    train: pandas.DataFrame
      Train split from dataset without NaN values.

    test: pandas.DataFrame
      Test split from dataset without NaN values.

    Returns
    -------
    train: pandas.DataFrame
      Train split from dataset with encoded categorical features.

    test: pandas.DataFrame
      Test split from dataset with encoded categorical features.
    """

    # categorical numerical list
    cat_num_features = train.select_dtypes("number").nunique()[
        train.select_dtypes("number").nunique() < 15
    ]
    cat_num_features_list = list(cat_num_features.index)
    cat_num_features_list.append("PROFESSION_CODE")

    # categorical features list
    cat_features = train.select_dtypes("O").nunique()
    cat_features_list = list(cat_features.index)

    # total categorical features
    encoding_features = [*cat_num_features_list, *cat_features_list]

    # remove features with lot of classes to perform ohe
    encoding_features.remove("CITY_OF_BIRTH")
    encoding_features.remove("RESIDENCIAL_CITY")
    encoding_features.remove("RESIDENCIAL_BOROUGH")

    # declare encoder with drop first
    ohe = OneHotEncoder(
        dtype=int, drop="first", sparse=False, handle_unknown="ignore"
    )
    ohe_fitted = ohe.fit(train[encoding_features])

    # encoding categorical columns on train and test split
    for feature in encoding_features:
        ohe_train = ohe.fit_transform(train[[feature]])
        ohe_test = ohe.transform(test[[feature]])
        # create columns with categories minus first on train and test
        categories = list(ohe.categories_[0][1:])
        categories_list = [feature + str(s) for s in categories]

        train[categories_list] = ohe_train
        test[categories_list] = ohe_test
        # drop original column on train and test
        train.drop(columns=feature, inplace=True)
        test.drop(columns=feature, inplace=True)

    # save fitted encoder to preprocess new data
    save_data_checkpoint(ohe_fitted, "../models/utils/ohe.pickle")

    return train, test
