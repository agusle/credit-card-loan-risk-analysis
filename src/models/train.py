"""Train model

This script allows the user to train a model from a selected dataset and save its results.

It uses several functions from other local modules with argparse input. 

Firstly, it loads and preprocess the selected the dataset. Then, it select which model to train and 
load the parameters grid from yaml file to perform hyperparameter tuning with RandomizedsearchCv 
if the user indicated that as an argument. 
In addition to doing that, the user can specify numbers of parameters sampled, number of
folds to perform cross-validation 
Finally, a score can be set as a minimun for the model to achieve in order to keep it.

Usage
-----
    python train.py /home/app/src/data/raw/PAKDD2010_Modeling_Data.txt lgbm yes 5 4 0.63

Returns
-------
    model : object
        Sklearn trained model.
"""

# Python standard libraries
import os
import argparse

# Third party libraries
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics

# Local libraries
from src import utils
from src.models import predict_model, hyper_grid
from src.features.preprocessing import preprocess_data


def parse_args():
    """
    Parse arguments written in command-line interface.
    """

    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument(
        "dataset_path",
        type=str,
        help=(
            "Full path to the file having all the applicants info. E.g. "
            "/home/app/src/data/raw/PAKDD2010_Modeling_Data.txt"
        ),
    )
    parser.add_argument(
        "model_name",
        type=str,
        help=(
            "Type of model to be trained. E.g."
            "logreg,rforest,lgbm,xgb,cbc,mlpc,voting"
        ),
    )
    parser.add_argument(
        "random_search",
        type=str,
        help=(
            "Load randomize search grid from hyperparameters.py. E.g."
            "Yes, No"
        ),
    )
    parser.add_argument(
        "n_iter",
        type=int,
        default=1,
        help=("Number of parameter settings that are sampled. E.g." "1,10,50"),
    )
    parser.add_argument(
        "cv",
        type=int,
        help=("Number of folds to perform cross-validation. E.g. " "2,3,4"),
    )
    parser.add_argument(
        "min_score",
        type=float,
        default=0.62,
        help=(
            "Minimum score to achieve for model saving. E.g." "0.60,0.62,0.64"
        ),
    )
    args = parser.parse_args()
    return args


def select_model(model_name):
    """
    Receives a Machine Learning model name and select it from others.

    Parameters
    ----------
    model_name : str
        Name for the ML model to be selected.

    Returns
    -------
    model = object
        ML model.
    """
    print(f"\nSelecting model: {model_name}\n")
    model_name = model_name.lower()

    if model_name == "logreg":
        model = LogisticRegression(random_state=42)
    elif model_name == "rforest":
        model = RandomForestClassifier(random_state=42)
    elif model_name == "lgbm":
        model = lgb.LGBMClassifier(random_state=42)
    elif model_name == "xgb":
        model = xgb.XGBClassifier(random_state=42)
    elif model_name == "cbc":
        model = CatBoostClassifier(random_state=42)
    elif model_name == "mlpc":
        model = MLPClassifier(random_state=42)
    elif model_name == "voting":
        clf1 = RandomForestClassifier(random_state=42)
        clf2 = lgb.LGBMClassifier(random_state=42)
        clf3 = xgb.XGBClassifier(random_state=42)
        clf4 = CatBoostClassifier(random_state=42)
        eclf = VotingClassifier(
            estimators=[
                ("rforest", clf1),
                ("lgbm", clf2),
                ("xgb", clf3),
                ("cbc", clf4),
            ],
            voting="soft",
        )
        model = eclf
    else:
        print(
            f'No name match with models. Please, try again and type one of the following:\n'
            f'"logreg,rforest,lgbm,xgb,cbc,mlpc,voting".'
        )

    return model


def load_grid(model_name):
    """
    Receives a model name and returns a grid of hyperparameters..

    Parameters
    ----------
    model_name : str
        Name of de model.

    Returns
    -------
    grid : Dict

    """
    model_name = model_name.lower()
    if model_name == "rforest":
        grid = hyper_grid.RANDOM_FOREST
    elif model_name == "lgbm":
        grid = hyper_grid.LIGHTGBM
    elif model_name == "xgb":
        grid = hyper_grid.XGBOOST
    elif model_name == "cbc":
        grid = hyper_grid.CATBOOST
    elif model_name == "mlpc":
        grid = hyper_grid.MLPC
    else:
        print("No grid was found")

    return grid


def save_model(model_name, model, roc_auc, min_score):
    """
    Saves the model as a pickle object if it achieves a minimum roc auc score.

    Parameters
    ----------
    model_name : str
        Name of the model.
    model : object
        Machine learning model.
    roc_auc : float

    """
    if roc_auc > min_score:
        print(
            f"\nTrained {model_name} exceeds {min_score} so its going to be saved\n"
        )
        roc_auc = round(roc_auc, 6)
        path = os.path.join(f"experiments/", f"{model_name}")
        if not os.path.exists(path):
            os.makedirs(path)
        utils.save_data_checkpoint(model, f"{path}/{model_name}_{roc_auc}")

    else:
        print(
            f"\nTrained {model_name} does not achieves a {min_score} as minimum score to be saved.\n"
        )


def get_roc_auc(model, y_test, features):
    """
    Calculates the roc_auc score metric.

    Parameters
    ----------
    model : object
        Scikit learn model.
    y_test : pandas.DataFrame
        Target values to test model.
    features:
        Features to predict target value.

    Returns
    -------
    roc_auc = float
        Metric to evaluate our model.
    """
    prob = model.predict_proba(features)
    pos_prob = prob[:, 1]
    fpr = metrics.roc_curve(y_test, pos_prob)[0]
    tpr = metrics.roc_curve(y_test, pos_prob)[1]
    roc_auc = metrics.auc(fpr, tpr)
    print(f"\nROC_AUC Score = {roc_auc:.6f}\n")
    return roc_auc


def train(dataset_path, model_name, random_search, n_iter, cv, min_score):

    print(f'\nPreprocessing {dataset_path.split("/")[-1]}\n')
    X_train, X_test, y_train, y_test = preprocess_data(dataset_path)

    model = select_model(model_name)

    if random_search.lower() == "yes":
        print(
            f"\nTuning {model_name} hyperparameters with Randomized Search\n"
        )
        randomcv = RandomizedSearchCV(
            estimator=model,
            param_distributions=load_grid(model_name),
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            verbose=5,
            random_state=42,
        )
        start_time = utils.timer(None)

        model = randomcv.fit(X_train, y_train)

        utils.timer(start_time)

        print(f"\nModel Best params:")
        for key, values in model.best_params_.items():
            print(f"{key} : {values}")
        print(f"\nBest Score: {model.best_score_:.6f}\n")

    else:
        start_time = utils.timer(None)
        print(f"\nTraining {model_name} without hyperparameters tuning\n")
        model.fit(
            X_train,
            y_train,
        )

        utils.timer(start_time)

    # get model performance
    y_pred = model.predict(X_test)
    predict_model.get_performance(y_pred, y_test, labels=[0, 1])
    roc_auc = get_roc_auc(model, y_test, X_test)

    save_model(model_name, model, roc_auc, min_score)

    return model


if __name__ == "__main__":
    args = parse_args()
    train(
        args.dataset_path,
        args.model_name,
        args.random_search,
        args.n_iter,
        args.cv,
        args.min_score,
    )
    print("\n------------Model trained successfully------------\n")
