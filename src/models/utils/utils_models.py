# Python Standard Libraries
import os
import pickle

# Third party Libraries
from datetime import datetime, date
import numpy as np
import pandas as pd
import xlrd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def save_data_checkpoint(filename, path):
    """
    Save picklable object to destination path.

    Parameters
    ----------
    filename : pickable obj
        name of the picklable objetc to save.

    path : str
        full path of destination directory.

    Returns
    -------
        Confirmation message.
    """
    if not os.path.exists(path):
        with open(path, "wb") as f:
            pickle.dump(filename, f, protocol=pickle.HIGHEST_PROTOCOL)
    return print(
        f"Object saved successfully in {path} with {np.round(os.path.getsize(path) / 1024 / 1024, 2)}MB."
    )


def load_data_checkpoint(path):
    """
    Load picklable object from destination path.

    Parameters
    ----------
    filename : str
        name of the picklable objetc to save.

    path : str
        full path of destination directory.

    Returns
    -------
        Confirmation message.
    """
    if os.path.exists(path):
        with open(path, "rb") as f:
            filename = pickle.load(f)
            print(f"Object loaded successfully from {path}.")
            return filename
    else:
        return print(f"Object or {path} does not exist.")


def clean_dataset(application):
    """
    Clean a dataset as pd.DataFrame from Constant, NaN and empty features.
    Firstly, it identifies the categorical, categorical numerical and numerical features
    that match any of the status mentioned. Finally, drop all identified features from dataframe.

    Parameters
    ----------
    application: pandas.DataFrame
      Raw application.

    Returns
    -------
    application: pandas.DataFrame
      Cleaned application.
    """

    # load raw dataset
    dataset = load_data_checkpoint("utils/X_train_raw.pickle")

    cat_features = dataset.select_dtypes("O").nunique()
    cat_num_features = dataset.select_dtypes("number").nunique()[
        dataset.select_dtypes("number").nunique() < 15
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
    empty_features = list(
        dataset.eq(" ").sum()[dataset.eq(" ").sum() > 25000].index
    )
    nan_features = list(
        dataset.isna().sum()[dataset.isna().sum() > 25000].index
    )

    # grouping lists in only one
    remove_features = [
        *nan_features,
        *empty_features,
        *catnum_constant,
        *catnum_constant_dist,
        *num_constant,
        *cat_constant,
    ]

    # remove selected features from application
    application = application.drop(columns=remove_features)

    return application


def outliers(application):
    """
    Replace outliers of following skewed distribution features
    by Numpy NaN values:

    ['OTHER_INCOMES','QUANT_DEPENDANTS', 'AGE']

    Fit features classes with delimited values on PAKDD2010_VariablesList.XLS
    replacing the incorrect ones with numpy.NaN.

    Parameters
    ----------
    application: pandas.DataFrame
      Single application.

    Returns
    -------
    application: pandas.DataFrame
      Single application with outliers converted to numpy.NaN.
    """

    # convert numeric columns from str
    application["OTHER_INCOMES"] = pd.to_numeric(
        application["OTHER_INCOMES"], downcast="float", errors="coerce"
    ).astype("int64")

    application["QUANT_DEPENDANTS"] = pd.to_numeric(
        application["QUANT_DEPENDANTS"], downcast="float", errors="coerce"
    ).astype("int64")

    application["AGE"] = pd.to_numeric(
        application["AGE"], downcast="float", errors="coerce"
    ).astype("int64")

    application["MARITAL_STATUS"] = pd.to_numeric(
        application["MARITAL_STATUS"], downcast="float", errors="coerce"
    ).astype("int64")

    application["RESIDENCE_TYPE"] = pd.to_numeric(
        application["RESIDENCE_TYPE"], downcast="float", errors="coerce"
    ).astype("int64")

    application["OCCUPATION_TYPE"] = pd.to_numeric(
        application["OCCUPATION_TYPE"], downcast="float", errors="coerce"
    ).astype("int64")

    # convert outliers to nan values
    application["OTHER_INCOMES"][
        application["OTHER_INCOMES"] > 190000.0
    ] = np.NaN
    application["QUANT_DEPENDANTS"][
        application["QUANT_DEPENDANTS"] > 20
    ] = np.NaN
    application["AGE"][application["AGE"] < 17] = np.NaN

    # adapt feature classes to PAKDD2010_VariablesList.XLS
    application["SEX"][application["SEX"] == "N"] = np.NaN
    application["STATE_OF_BIRTH"][
        application["STATE_OF_BIRTH"] == "XX"
    ] = np.NaN
    application["MARITAL_STATUS"][application["MARITAL_STATUS"] == 0] = np.NaN
    application["RESIDENCE_TYPE"][application["RESIDENCE_TYPE"] == 0] = np.NaN
    application["OCCUPATION_TYPE"][
        application["OCCUPATION_TYPE"] == 0
    ] = np.NaN

    return application


def impute_nan(application):
    """
    This function replace NaN values depending on feature dtype:

    Numerical feature (number): replace with feature's median.
    Categorical feature (object): replace with feature's mode.

    Parameters
    ----------
    application: pandas.DataFrame
      Single application with outliers replaced with NaN.

    Returns
    -------
    application: pandas.DataFrame
      Single application with NaN values replaced.
    """

    # load raw dataset
    dataset = load_data_checkpoint("utils/X_train_interim.pickle")

    # Replace empty string values with np.NAN.
    application = application.replace(" ", np.NaN)
    # filter numeric columns a dataframe with sum of NaN values of numerical features
    num_columns = dataset.select_dtypes(include="number").columns

    # input median values for all numerical columns with missing data
    for feature in num_columns:
        if str(application[feature]) == "nan":
            application[feature] = dataset[feature].median()

    # filter object columns a dataframe with sum of NaN values of numerical features
    cat_columns = dataset.select_dtypes(include="O").columns

    # input mode values for all non numerical columns with missing data
    for feature in num_columns:
        if str(application[feature]) == "nan":
            application[feature] = dataset[feature].mode().loc[0]

    return application


def scaling(application):
    """
    Scale numerical features using Sklearn StandardScaler.
    Firstly, create a list of numerical features without categorical numerical ones.

    Parameters
    ----------
    application: pandas.DataFrame
      Single application without NaN values.

    Returns
    -------
    application: pandas.DataFrame
      Single application with scaled numerical features.
    """

    dataset = load_data_checkpoint("utils/X_train_interim.pickle")
    # create a list of features we want to scale
    cat_num_features = dataset.select_dtypes("number").nunique()[
        dataset.select_dtypes("number").nunique() < 15
    ]

    number_dtype = dataset.select_dtypes("number").columns
    num_features_list = list(set(number_dtype) - set(cat_num_features.index))
    num_features_list.remove("PROFESSION_CODE")

    # load fitted standard scaler
    scaler = load_data_checkpoint("utils/scaler.pickle")

    # scale column by column with each corresponding fit between train and test
    for feature in num_features_list:
        application[feature] = scaler.transform(
            application[feature].values.reshape(1, -1)
        )

    return application


def encode(application):
    """
    Encode categorical features using sklearn One Hot Encoder.
    It ignore following features with more than 1k classes dropping those from datasets:
    ['CITY_OF_BIRTH','RESIDENCIAL_CITY','RESIDENCIAL_BOROUGH']

    The encoder is used dropping the first column.

    Parameters
    ----------
    application: pandas.DataFrame
      Single application without NaN values.

    Returns
    -------
    application: pandas.DataFrame
      Single application with encoded categorical features.
    """

    dataset = load_data_checkpoint("utils/X_train_interim.pickle")

    # categorical numerical list
    cat_num_features = dataset.select_dtypes("number").nunique()[
        dataset.select_dtypes("number").nunique() < 15
    ]
    cat_num_features_list = list(cat_num_features.index)
    cat_num_features_list.append("PROFESSION_CODE")

    # categorical features list
    cat_features = dataset.select_dtypes("O").nunique()
    cat_features_list = list(cat_features.index)
    # total categorical features
    encoding_features = [*cat_num_features_list, *cat_features_list]

    # remove features with lot of classes to perform ohe
    encoding_features.remove("CITY_OF_BIRTH")
    encoding_features.remove("RESIDENCIAL_CITY")
    encoding_features.remove("RESIDENCIAL_BOROUGH")

    # load encoder
    ohe = load_data_checkpoint("utils/ohe.pickle")

    # create dataframe with encoded features
    enc_df = pd.DataFrame(
        ohe.transform(application[encoding_features].values.reshape(1, -1))
    )
    # concatenate application series with encoded dataframe
    application = application.append(enc_df.iloc[0])
    # drop original columns
    application = application.drop(columns=encoding_features)

    return application
