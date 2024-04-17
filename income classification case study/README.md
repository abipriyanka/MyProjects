# CLASSIFYING PERSONAL INCOME 

## Objective:
The objective is to simplify the data system by reducing the number of variables to be studied, without sacrificing too much of accuracy. Such a system would help Subsidy Inc. in planning subsidy outlay, monitoring, and preventing misuse.

### Required packages:
- os
- pandas as pd
- numpy as np
- seaborn as sns
- train_test_split from sklearn.model_selection
- KNeighborsClassifier from sklearn.neighbors
- accuracy_score and confusion_matrix from sklearn.metrics

### Importing data:
The data is imported from the 'income.csv' file. 

### Data pre-processing:
1. Identified missing values in the columns 'Jobtype' and 'Occupation'.
2. Dropped rows with missing values.
3. Reindexed the salary status names to 0 and 1.
4. Converted categorical variables into dummy/indicator variables.
5. Separated the input and output variables.

### KNN Classification:
- Used K Nearest Neighbors (KNN) classifier.
- Split the data into training and testing sets.
- Fitted the model with the training data.
- Predicted the test values.
- Evaluated the performance using confusion matrix and accuracy score.
- Analyzed the effect of different K values on classifier performance.

### License:
This project is licensed under the MIT License.
