                                                                                        Credit risk analysis 
                                                                                        using deep learning

                                                                                        15.07.22
                                                                                        Agustín Leperini
                                                                            
# <center> Exploratory Data Analysis Report (EDA) </center>

## Introduction

The dataset was created for a famous competition called [PAKDD 2010 (Pacific-Asia Knowledge Discovery and Data Mining)](https://pakdd.org/archive/pakdd2010/) and the objective was a Re-Calibration of a Credit Risk Assessment System Based on Biased Data. 
 
There were 3 datasets used for the Challenge. They were collected during period from 2006 to 
2009,  and  came  from  a  private  label  credit  card  operation  of  a  Brazilian  credit  company 
and its partner shops.  Those mentioned dataset were downloaded from a s3 bucket hosted in [Amazon Simple Storage Service](https://aws.amazon.com/s3/) through [dataset_download.py](https://github.com/agusle/final-project/blob/main/src/dataset/dataset_download.py).
 
The  prediction  targets  to  detect  are  the  “bad”  clients.  A  client  is  labeled  as  “bad”  (target 
variable=1) if he/she made 60 days delay in any payment of the bills contracted along the first 
year  after  the  credit  has  been  granted.  In  short,  the  clients  that  do  not  pay  their  debt  are 
labeled as “bad”. 

## Comparative table between dataset files 

In the following table I will compare the 3 different downloaded files to analyze the potential usefulness of each for future model training:


|                          |                     PAKDD2010_Modeling_Data.txt                    |                     PAKDD2010_Leaderboard_Data.txt                    |                       PAKDD2010_Prediction_Data.txt                       |
|--------------------------|:-----------------------------------------------:|:--------------------------------------------------:|:------------------------------------------------------:|
| Applicants               |                      50.000                      |                        20.000                       |                          20.000                         |
| Client ID                |                      1-50k                      |                    50.001 - 70k                    |                       70.001 -90k                      |
| % Total applicants             |                      55,55%                     |                       22,22%                       |                         22,22%                         |
| Features                 |                        54                       |                         53                         |                           53                           |
|  Categorical           |                        20                       |                         20                         |                           18                           |
|  Categorical numerical |                        26                       |                         25                         |                           25                           |
|  Numerical             |                        8                        |                          8                         |                            8                           |
| Data with missing values |                      88%                     |                          0%                         |                            0%                           |
| Labels  0 - GOOD         |                 36959  - 73.92 %                |         PAKDD2010_Leaderboard_Submission_Example.txt         |                            0                           |
| Labels 1 - BAD           |                 13041  - 26.08 %                |         PAKDD2010_Leaderboard_Submission_Example.txt         |                            0                           |
| Utility         |                      Train/Validation/Test                      |                     **Desestimated**                     |                          **Desestimated**                          |
| Data exploratory details | [1.0 - avl - Initial_data_exploration - Modeling](https://github.com/agusle/final-project/blob/main/notebooks/1.0-avl-Initial_data_exploration-Modeling.ipynb) | [1.1 - avl - Initial_data_exploration - Leaderboard](https://github.com/agusle/final-project/blob/main/notebooks/1.1-avl-Initial_data_exploration-Leaderboard.ipynb) | [1.2 - avl - Initial_data_exploration - Prediction-data](https://github.com/agusle/final-project/blob/main/notebooks/1.2-avl-Initial_data_exploration-Prediction-data.ipynb) |

## Dataset conclusions 

1 - Noticed that **PAKDD2010_Leaderboard_Data.txt** has 20k applicants inputs without target, but I found that **Leaderboard_Submission_Example.txt** target probability comes from competition to give instantaneous feedback about the accuracy of the different trained models byteams. The question I had was: 
- Should I use that probability mentioned before to test the model knowing there aren't real labels because was a feedback of an instance of a ML model developed by a competition team?

**I believed I didn't have to use it because I maybe introducing biased information into an already biased dataset. So, I should depeen our data analysis on PAKDD2010_Modeling_Data only.**

2 - The last file **PAKDD2010_Prediction_Data.txt** had no labels at all so it made it impossible to build a supervised machine or deep learning model.

3 - I had the challenge of working with a very small dataset. From a total of 90.000 aplicants, only the 55,55% of it was useful.

## Feature engineering

Once I focused on the file I wanted to work on to train the model I came to the following conclusion by doing feature engineering on the aforementioned [notebook](https://github.com/agusle/final-project/blob/main/notebooks/1.0-avl-Initial_data_exploration-Modeling.ipynb). Through that analysis I was able to conclude that:

- **FIRST THINGS FIRST**: From the total of 54 feature I realized that on of them was the target label and the other the applicant ID. Thus, the latter was used as an index and the former was analyzed and then separated by splitting the dataset.
- **FROM 52 to 32 FEATURES:** There were 4 features where more than 50% of null values were found. Also, there were 2 features where more than 60% of empty values were found and finally, other 14 features had constant values (above 99,3% of same value). All those features were discarded for model training.
- **MISSING VALUES**: Of the remaining 32 features there were 4 features where missing values accounted less than 20% and other 5 different features where empty values accounted for the same percentage. All those values were replaced by the mode or median, depending on the type of characteristic to which they belong (categorical or numerical).
- **FEATURES VALUES UNMATCHED WITH VARIABLES LIST:** As you'll find on the dataset there was an excel file called PAKDD2010_VariablesList.XLS containing the possible values for each feature in the dataset. There were 6 features (3 categorical and 3 categorical numerical) that did not follow that order so I managed to impute those unexpected values with the feature mode.
- **OUTLIERS**: 3 features were selected to remove outliers:
    - QUANT_DEPENDANTS > 20 which is the number of people who are economically dependent of the applicant. 
    - AGE < 17 because applicants are not legally allowed to apply for a loan unless they have special permits. 
    - OTHER_INCOMES > 190000 that represent monthly income from activities other than applicant's regular job. 
- **IGNORED FEATURES**: The following features were discarded for preprocessing because they have too much classes to be encoded: 
    - CITY_OF_BIRTH: 9910 
    - RESIDENCIAL_CITY: 3529 
    - RESIDENCIAL_BOROUGH: 14511 
- **BONUS:** There was one really interesting feature (AGE) in wich the distribution of the target value increased substantially in the range between 17 and 20 years. It went from an average of 26% to 40%. Therefore,the model is very sensitive to this feature.

## Split dataset
Due to the distribution of the target label, the dataset was **split in a stratified manner** to mantain proportions of the target labels in the returned datasets.

## Preprocessing
All the feature enginneering and split stages of data analyisis were carried out by [preprocessing.py](https://github.com/agusle/final-project/blob/main/src/features/preprocessing.py) to prepare data for model training with scaling, encoding


## Personal productivity
- **High level tasks**: 5 
    - Download Dataset
    - Surfing web to understand dataset background
    - EDA notebooks
    - Feature Engineering
    - Preprocessing script
- **Total time**: 44:45hs about one week of work.
- **Percentage of total project workload**: 30%.

## What can be improved
Dataset
- Obtain more applications to dataset it's always useful.Also, try to include from other sources external data like: 
     - Transactional Information, 
     - Social Media, 
     - Other third party information.
- Add [bz2](https://docs.python.org/3/library/bz2.html) python library to compress the space needed when storing different instances of data according to their processing.
- Try different dimensionality reduction methods to include ignored features into model consideration it'll surely increase score.
