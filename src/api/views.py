# Third party libraries
from flask import Blueprint, render_template, request, jsonify, flash
import csv
import json
import uuid

# Local Libraries
from middleware import model_predict
import settings

router = Blueprint("app_router", __name__, template_folder="templates")


@router.route("/", methods=["GET"])
def index():
    """
    Render the index.html where the application form is.

    Returns
    -------
    Render the index.html
    """

    return render_template("form.html")


@router.route("/application", methods=["POST"])
def application():
    """
    Receives the application form and convert its values into a dict. Then send the dict
    to redis through model_predict() and wait for a response. Finally renderize the score in
    application_success.html.


    Returns
    -------
    prediction, prediction_proba : tuple(float, float)
        Render the application_success.html with the model score.
    """

    # receive request form and transform to dictionary
    req = request.form.to_dict(flat=False)

    application_dict = {}
    for element in req:
        application_dict[element] = req[element][0]

    # encole dictionary to a redis job
    pred = model_predict(application_dict)
    context = {
        "label": pred[0],
        "score": pred[1],
    }

    # store application info in new applicants submissions file
    new_application_path = settings.UPLOADS_FILEPATH

    with open(new_application_path, "a+") as f:
        writer = csv.DictWriter(f, application_dict.keys(), delimiter="\t")
        writer.writerow(application_dict)

    # handle html responses by predictions
    if context["label"] == 1:
        return render_template("fail.html", result=context)
    elif context["label"] == 0:
        return render_template("success.html", result=context)


@router.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint used to get predictions without need to access the UI.

    Parameters
    ----------
    req : dict
        Input dict we want to get predictions from.

    Returns
    -------
    flask.Response
        JSON response from our API having the following format:
            {
                "success": bool,
                "label": int,
                "score": float,
            }

        - "success" will be True if the input file is valid and we get a
          prediction from our ML model.
        - "label" model predicted target as int.
        - "score" model confidence score for the predicted target as float.
    """

    try:
        rpse = {"success": False, "label": None, "score": None}
        # Request object is a json.
        req = request.get_json()
        pred = model_predict(req)
        rpse = {"success": True, "label": pred[0], "score": pred[1]}
        return jsonify(rpse), 200
    except:
        return jsonify(rpse), 400
