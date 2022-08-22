<p align="center" width="100%">
    <img src="https://github.com/agusle/credit-risk-analysis-using-deep-learning/blob/main/img/project-logo.PNG">
</p>

<p align="center">
    <a href="https://github.com/agusle/credit-risk-analysis-using-deep-learning/commits/main">
    <img src="https://img.shields.io/github/last-commit/agusle/credit-risk-analysis-using-deep-learning?logo=Github"
         alt="GitHub last commit">
    <a href="https://github.com/agusle/credit-risk-analysis-using-deep-learning/issues">
    <img src="https://img.shields.io/github/issues-raw/agusle/credit-risk-analysis-using-deep-learning?logo=Github"
         alt="GitHub issues">
    <a href="https://github.com/agusle/credit-risk-analysis-using-deep-learning/pulls">
    <img src="https://img.shields.io/github/issues-pr-raw/agusle/credit-risk-analysis-using-deep-learning?logo=Github"
         alt="GitHub pull requests">
</p>

<p align="center">
  <a href="#-about">About</a> •
  <a href="#-lets-play">Let´s play</a> •
  <a href="#-architecture">Architecture</a> • 
  <a href="#%EF%B8%8F-install-and-run">Install and Run</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-contribute">Contribute</a> •
</p>

------------------

## 📖 About
**Problem**: Binary classification with the aim of predicting risky credit operations.

In this particularly case I classified **if a credit applicant is creditworthy** analyizing its personal and  financial data to predict if is going to repay a loan on time. It is considered that an applicant have defaulted on its debt if it does not pay within 60 days of the due date so in this case will label with 0 those that are solvent and with 1 those that are not.

**Industries**: fintech, banks, logistics, delivery apps, insurtech, many others related. 

**Solution**:
Determine whether a new loan applicant will be able to repay its debt or not. Very small dataset of 50,000 applicants. Trained many machine learning models achieving +0.64 ROC AUC and compared results against a deep learning model like Multilayer perceptron. Deployed in AWS instances using Docker and also using
API-based web-service application with Flask.
You can see detailed information in the following **reports**:
 - [Exploratory Data Analysis](https://github.com/agusle/credit-risk-analysis-using-deep-learning/blob/main/reports/Data_exploratory_report.md)
 - [Modeling](https://github.com/agusle/credit-risk-analysis-using-deep-learning/blob/main/reports/Modeling_report.md)

------------------

## 🎮 Let´s play
![project-demo](img/project-demo.gif)

------------------

## 👷🏿 Architecture
API service that is backed by a scikit learn model. Basically, through a form used as an interface between our service and the user, who will enter data that will be processed as an array with 52 features, we will receive two possible answers, each with 2 results that will be the following: a label of 0 or 1 and a probability from 0 to 100 that the label is 1. 

![project-archiquetcture](img/project_architecture.drawio.png)

### **So, how does it works?**  
It uses *Flask* and *Redis* to queue tasks to be processed by the machine learning model so the communication between our services (*api* and *model*) will be done using *Redis*. Every time *api* wants to process a loan application, it will send the data through Redis to the *model* service. Then, the *model* get predictions and return the results back to *api*. 

In addition, I used *docker* to make the application portable and sharable, using less infrastructure resources and ready to be deployed in production. In this case, I had to orchestrate between several services technically called containers, using *docker compose*.  Finally, I used *Amazon Web Services* to gather data an train the models. 

The API has 2 endpoints and an additional one to connect directly avoiding the use of the interface.

------------------

## ⚡️ Install and Run 

Listed below you'll find an example or application usage to run the services using compose:

- **To run the services:** 

```bash
$ cd docker/
$ docker-compose up --build
```

- **To stop the services:**

```bash
$ docker-compose down
```
------------------

## 👀 Usage
Open your internet browser and type:

```bash
http://localhost/
```
------------------

## 👍 Contribute
Check out [`CONTRIBUTING.md`](https://github.com/agusle/credit-risk-analysis-using-deep-learning/blob/main/CONTRIBUTING.md)