# Python standard Library
import time
import pickle
import redis
import json
import os

# Third party libraries
import pandas as pd
import lightgbm as lgb

# Local Libraries
from utils import utils_models
import settings

# connect to Redis
db = redis.Redis(
    host=settings.REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB_ID
)
# load trained model
model = utils_models.load_data_checkpoint("utils/best_LGB.pickle")


def preprocess(application):
    """
    Get model prediction from an input data

    Parameters
    ----------
    sigle_applicant : pandas.DataFrame
        DataFrame Series with applicant's form information submitted

    Returns
    -------
    class_name, pred_probability : tuple(int, float)
        Model predicted class as an integer [0 or 1] and the corresponding confidence
        score as a number.
    """

    application_prepro = utils_models.clean_dataset(application)
    application_prepro = utils_models.outliers(application_prepro)
    application_prepro = utils_models.impute_nan(application_prepro)
    application_prepro = utils_models.scaling(application_prepro)
    application_prepro = utils_models.encode(application_prepro)

    # ignore features with many classes to perform preprocessing
    ignored_features = [
        "CITY_OF_BIRTH",
        "RESIDENCIAL_CITY",
        "RESIDENCIAL_BOROUGH",
    ]
    application_prepro = application_prepro.drop(columns=ignored_features)

    return application_prepro


def predict(application):
    """
    Get model prediction from an input data

    Parameters
    ----------
    sigle_applicant : pandas.DataFrame
        DataFrame Series with applicant's form information submitted

    Returns
    -------
    class_name, pred_probability : tuple(int, float)
        Model predicted class as an integer [0 or 1] and the corresponding confidence
        score as a number.
    """

    application_prepro = preprocess(application)

    # make predictions with loaded trained model
    label = model.predict(application_prepro)[0]
    score = model.predict_proba(application_prepro)[:, 1][0]

    return tuple([label, round(float(score) * 100, 2)])


def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.
    """

    while True:
        # take a new job from Redis
        _, msg = db.brpop(settings.REDIS_QUEUE)
        if msg:
            # unpack data from job
            msg_dict = json.loads(msg)
            application_dict = msg_dict["application"]
            # transform to dataframe an save application
            application = list(application_dict.values())
            features_list = utils_models.load_data_checkpoint(
                "utils/features_list.pickle"
            )
            application = pd.DataFrame([application], columns=features_list)

            # make prediction
            pred_label, pred_score = predict(application)
            # create dictionary and json object with results
            output_msg = {
                "label": int(pred_label),
                "score": float(pred_score),
            }
            output_data = json.dumps(output_msg)
            # store the results on Redis using the original job ID as key
            db.set(msg_dict["id"], output_data)
            # sleep between loops
            time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    print("------------Launching ML service------------")
    classify_process()
