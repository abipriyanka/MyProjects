# Bank Marketing Classification

This project aims to predict whether a client will subscribe to a term deposit or not, based on various features such as age, job, marital status, education, etc. The dataset used for this analysis is the Bank Marketing dataset.

## Description

The dataset contains information about clients contacted in a marketing campaign for a bank's term deposit. The goal is to build a classification model to predict whether a client will subscribe to the term deposit ('yes' or 'no').

## Introduction

The project involves the following steps:

1. Data loading and exploration.
2. Data preprocessing including handling missing values, encoding categorical variables, and scaling numerical features.
3. Building classification models using Logistic Regression, K-Nearest Neighbors, Decision Tree, and Random Forest algorithms.
4. Evaluating model performance using accuracy score, confusion matrix, and ROC curve analysis.

## Data Exploration
The dataset has been explore to understand its structure, features, and distributions. The relationship between different variables and the target variable ('deposit') has been analyzed.

## Preprocessing
Data preprocessing involves handling missing values, encoding categorical variables, and scaling numerical features. Additionally, the dataset is split into training and testing sets for model building and evaluation.

## Classification Models

### Logistic Regression
- Fitted logistic regression model to the data.
- Evaluated model performance using confusion matrix and accuracy score.
- Plotted the ROC curve and calculated the AUC ROC score.

### K-Nearest Neighbors (KNN)
- Trained the KNN classifier with k=5.
- Assessed model performance using confusion matrix and accuracy score.
- Visualized the ROC curve and computed the AUC ROC score.

### Decision Tree
- Built a decision tree classifier.
- Examined model performance using confusion matrix and accuracy score.
- Generated the ROC curve and determined the AUC ROC score.

### Random Forest
- Constructed a Random Forest classifier with 30 estimators and max depth of 10.
- Measured model performance using confusion matrix and accuracy score.
- Plotted the ROC curve and calculated the AUC ROC score.

## Model Comparison

| Model                | Accuracy Score |
|----------------------|----------------|
| Logistic Regression  | 75.45%         |
| K-Nearest Neighbors  | 76.64%         |
| Decision Tree        | 76.27%         |
| Random Forest        | 82.92%         |

## Evaluation
Random Forest has demonstrated the highest AUC ROC Score and accuracy score compared to other models. Therefore, Random Forest emerges as the best classifier for the given dataset. It exhibits superior performance in predicting whether a client will subscribe to the term deposit or not.

## License
This project is licensed under the MIT License.
