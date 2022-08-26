                                                                                        Credit risk analysis 
                                                                                        using deep learning

                                                                                        29.07.22
                                                                                        Agustín Leperini

# <center> Model Evaluation Report</center>
## Model evaluation metric 
As I am facing a binary classification problem, the **AUC-ROC Curve** is a performance measure for classification problems at various threshold settings. 

ROC is a probability curve and AUC represents the degree or measure of separability. Indicates how much
the model is able to distinguish between classes. The higher the AUC, the better the model is at predicting 0
classes as **GOOD CLIENT** and 1 classes as **BAD CLIENT**. By analogy, the higher the AUC, the better the model.

This method is convenient for the following 2 reasons:

- It is scale invariant,**it measures how well the predictions are ranked**, rather than their absolute values.
- It is invariant with respect to the classification threshold, **it measures the quality of the model predictions, regardless of which classification threshold is chosen**.


## Has anyone solved it before?
Yes, it is not advisable to always try to reinvent the wheel so I managed to surf the web until I found the following competitions:

|                        | PAKDD Competition | Kaggle Competition |
|------------------------|-------------------|--------------------|
| Carried out            |        2010       |        2016        |
| Top 5 scores (ROC_AUC) |   0.645 - 0.637   |    0.690 - 0.687   |
| Link                   |        [Here](https://pakdd.org/archive/pakdd2010/)       |        [Here](https://www.kaggle.com/competitions/pakdd2010-dataset/leaderboard)        |



## Training model
In this evaluation report I'm comparing the results of several experiments carried out with the following models:
- Random Forest ------------> **Machine Learning**
- Boosting -------------------> **Machine Learning**
- Multilayer Perceptron------> **Deep Learning** 

During this iterative process using AWS Elastic Computing (EC2) cloud server I evaluate the trade-off between the resources and time consumption with the accuracy we desire, to choose the best approach between the mentioned techniques.

You will find the script for model training in [train.py](https://github.com/agusle/credit-risk-analysis-using-deep-learning/blob/main/src/models/train.py) with the corrpesponding documentation.


## Hardware specifications of server used for training:
### GPU:
NVIDIA-SMI 470.129.06  
Driver Version: 470.129.06  
CUDA Version: 11.4 
Model: Tesla K80 
Memory: 11.441 MiB 

## Models training performance and metrics
<p align="center">
    <img src="https://github.com/agusle/credit-risk-analysis-using-deep-learning/blob/main/img/models.png">
</p>

### Baseline
Beginning the analysis with a simple baseline by trying to get a basic summary of the performance of the different models. In this case, XGBoost seemed to perform better than the rest.

|                     | Random Forest |          LightGBM |            XGBoost |          CatBoost |
|--------------------:|--------------:|------------------:|-------------------:|------------------:|
|            Accuracy |        0.7364 |            0.7395 |         **0.7397** |            0.7392 |
|             ROC AUC |      0.615017 |          0.630551 |       **0.639458** |          0.616246 |
|       Training Time | 0 min 37 secs | 2 mins and 45 secs | 8 mins and 58 secs | **0 min 21 secs** |
| Inference time Unit | 0.927343 secs | **0.004546 secs** |      0.019042 secs |     0.549788 secs |
|                Size |      190.11MB |            0.72MB |             0.79MB |         **0.2MB** |

- Notebook reference: [2.0-avl-Baseline_Model_evaluation_1.ipynb](https://github.com/agusle/credit-risk-analysis-using-deep-learning/blob/main/notebooks/2.0-avl-Baseline_Model_evaluation_1.ipynb)

### Hyperparameters Tuning
After baseline stage I went deeper to hyperparameters tuning to improve metrics and also try to compare best machine learning models vs a deep learning Multilayer perceptron. In this case, best overall model seemed to be LightGBM.

|                    |                 Random Forest |            LightGBM |                      XGBoost |           CatBoost |                MLP Classifier |
|-------------------:|------------------------------:|--------------------:|-----------------------------:|-------------------:|------------------------------:|
|           Accuracy |                        0.7395 |              **0.7411** |                        0.395 |             0.7395 |                        0.7393 |
|            ROC AUC |                      0.642177 |           **0.643960** |                     0.642581 |           0.633388 |                      0.634936 |
| Tuning improvement |                       **+ 0.027** |             + 0.013 |                      + 0.003 |            + 0.017 |               -               |
|        Tuning Time | 2 hours , 49 mins and 45 secs | 11 mins and 10 secs | 4 hours , 8 mins and 41 secs | **9 mins and 10 secs** | 4 hours , 20 mins and 44 secs |
|     Inference Time |                 0.048623 secs |       **0.006243 secs** |                0.007087 secs |      0.013457 secs |                 0.009351 secs |
|               Size |                       45.34MB |              0.45MB |                       0.79MB |              **0.2MB** |                        1.56MB |

- Notebook reference: [2.1-avl-Tuned_Model_evaluation_2.ipynb](https://github.com/agusle/credit-risk-analysis-using-deep-learning/blob/main/notebooks/2.1-avl-Tuned_Model_evaluation_2.ipynb)

### New ensemble methods and final analysis
Finally, I spent time creating ensemble techniques that combines all created models to see if metrics improvement but it did not change results significantly.

|                |                 Random Forest |            LightGBM |                      XGBoost |           CatBoost |                MLP Classifier | Ensemble soft voting |   Ensemble stacked |
|---------------:|------------------------------:|--------------------:|-----------------------------:|-------------------:|------------------------------:|---------------------:|-------------------:|
|       Accuracy |                        0.7395 |              0.7411 |                        0.395 |             0.7395 |                        0.7393 |               **0.7421** |             0.7409 |
|        ROC AUC |                      0.642177 |            0.643960 |                     0.642581 |           0.633388 |                      0.634936 |             0.649130 |           **0.649733** |
|    Tuning Time | 2 hours , 49 mins and 45 secs | **11 mins and 10 secs** | 4 hours , 8 mins and 41 secs | 9 mins and 10 secs | 4 hours , 58 mins and 28 secs |                    - |                  - |
|  Training Time |                       39 secs |              14 secs |                      59 secs |             **10 secs** |             36 mins and 29 secs |  2 mins and 33 secs | 9 mins and 14 secs |
| Inference Time |                 0.048623 secs |       0.006243 secs |                0.007087 secs |      0.013457 secs |                **0.003036 secs** |        0.331205 secs |      0.399165 secs |
|           Size |                       45.34MB |              0.44MB |                       **0.19MB** |             0.91MB |                        1.56MB |              47.11MB |            47.11MB |

- Notebook reference: [2.2-avl-Final_Model_evaluation.ipynb](https://github.com/agusle/credit-risk-analysis-using-deep-learning/blob/main/notebooks/2.2-avl-Final_Model_evaluation.ipynb)

## Conclusion
After all the training, hyperparameter tuning and trying new ensemble learning methods I analyzed what model I wanted to predict the results for each new applicant based on the following 2 metrics:
- **AUC (Area under the ROC Curve)** is as the probability that the model ranks a random positive example more highly than a random negative example.
- **Inference time** is the amount of time it takes for a machine learning model to process new data and make a prediction.

Summarizing the results I made a podium with the performance of all models.

| Metrics / Model | Random Forest | LightGBM | XGBoost | CatBoost | MLP Classifier | Ensemble soft voting | Ensemble stacked |
|-----------------|---------------|----------|---------|----------|----------------|----------------------|------------------|
| AUC             |               |     🥉    |         |          |                |           🥈          |         🥇        |
| Inference time  |               |     🥈    |    🥉    |          |        🥇       |                      |                  |


Besides [Ensemble Stacked](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization) achieved best AUC score (almost 0.65) and [MLP Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) the best inference time, it is more advisable to choose a model that has a balance between these two metrics, so the model **I selected to predict results is [LightGBM](https://lightgbm.readthedocs.io/en/v3.3.2/).**

- During the evaluation of all experiments I can conclude that a trained LightGBM model reached and acceptable level of prediction.
- With a short amount of time I reached the **top position of PAKDD 2010 competition and top 20 in Kaggle.**
- Sometimes less is more and complex models could not be always the best solution. During the experimentation **Boosting models were the best performing ones.**

## Personal productivity
- **High level tasks**: 3 
    - Modeling (Define models to use and build a script for training)
    - Set up AWS server.
    - Model results comparison.
- **Total time**: 33hs about one week of work.
- **Percentage of total project workload**: 25%.

## What can be improved
- Build a Multilayer Perceptron with Tensorflow and deepen into its hyperparameter configuration.
- Improve train script to bring the user the possibility to load all configuration from an unique .yaml file.
- Use configuration variables to load the model I am going to use when running the application. The model is actually hardcoded into the source code.
- Load the models from a cloud service storage.
- Make the model results more explainable to the end user.
- Add test to train scripts and configurations file validations.
