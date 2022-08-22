# Third party libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


def get_performance(predictions, y_test, labels):
    """
    Calculates several sklearn metrics of model's perfomance.
    Print metrics, classification report and confusion matrix.

    Parameters
    ----------
    y_test : pandas.DataFrame
        Target values to test model.
    predictions : numpy.array
        Prediction made by the model.
    labels : list
        Labels of prediction.

    Returns
    -------
    accuracy = float
    precision = float
    recall = float
    f1_score = float
    """
    accuracy = metrics.accuracy_score(y_test, predictions)
    precision = metrics.precision_score(y_test, predictions)
    recall = metrics.recall_score(y_test, predictions)
    f1_score = metrics.f1_score(y_test, predictions)

    report = metrics.classification_report(y_test, predictions, labels=labels)

    cm = metrics.confusion_matrix(y_test, predictions, labels=labels)
    cm_as_dataframe = pd.DataFrame(data=cm)

    print("Model Performance metrics:")
    print("-" * 30)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)
    print("\nModel Classification report:")
    print("-" * 30)
    print(report)
    print("\nPrediction Confusion Matrix:")
    print("-" * 30)
    print(cm_as_dataframe)

    return accuracy, precision, recall, f1_score


def plot_roc(model, y_test, features):
    """
    Calculates the roc auc metrics and plot roc curve.

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

    plt.figure(figsize=(10, 5))
    plt.plot(
        fpr, tpr, label=f"ROC curve (area = {roc_auc:.6f})", linewidth=2.5
    )
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc
