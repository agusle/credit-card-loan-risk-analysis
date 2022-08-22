# Python standard Library
import time
import uuid
import json

# Third party libraries
import redis

# Local Libraries
import settings

db = redis.Redis(
    host=settings.REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB_ID
)


def model_predict(application):
    """
    Receives an application and queues the job into Redis.
    Will loop until getting the answer from our ML service.

    Parameters
    ----------
    application : dict
        dict with features values.

    Returns
    -------
    prediction, prediction_proba : tuple(int, float)
        Model predicted target as a integer [0 or 1] and the corresponding confidence
        score as a number.
    """

    job_id = str(uuid.uuid4())
    job_data = {
        "id": job_id,
        "application": application,
    }
    job_data = json.dumps(job_data)

    db.rpush(settings.REDIS_QUEUE, job_data)

    # Loop until we received the response from our ML model
    while True:
        if db.exists(job_id):
            # get job of redis queue
            prediction_dict = json.loads(db.get(job_id))
            # delete the job from Redis after we get the results
            db.delete(job_id)
            break
        # Sleep some time waiting for model results
        time.sleep(settings.API_SLEEP)

    return tuple([prediction_dict["label"], prediction_dict["score"]])
