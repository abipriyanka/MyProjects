#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data analysis tools
import pandas as pd
import numpy as np

# Data Visualization Tools
import seaborn as sns
import matplotlib.pyplot as plt

# Data Pre-Processing Libraries
from sklearn.preprocessing import LabelEncoder,StandardScaler

# For Train-Test Split
from sklearn.model_selection import train_test_split

# Libraries for various Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Metrics Tools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
#For Receiver Operating Characteristic (ROC)
from sklearn.metrics import roc_curve ,roc_auc_score, auc


# In[2]:


import os


# In[3]:


#setting the directory
os.chdir("D:\C Backup\Desktop\datasets")


# In[4]:


data=pd.read_csv("bank_marketing (1) (2).csv")


# In[5]:


data


# In[6]:


data1 = data.drop(data.columns[0],axis=1)
print(data1)


# In[7]:


data1


# In[8]:


#information about the data
data1.info()


# In[9]:


#checking for null values
data.isnull().sum()


# In[10]:


data.isna().sum()


# In[11]:


#checking for duplicate values
print(data.duplicated().value_counts())


# In[12]:


data1.describe()


# In[13]:


data1.select_dtypes(include=object)


# In[14]:


#Summary of categorical variables
summary_cate = data1.describe(include = "O")
print(summary_cate)


# In[15]:


#count of unique values
data1_unique=data1.nunique().to_frame().reset_index()


# In[16]:


data1_unique


# In[17]:


#printing the unique values for each categorical variables
print('Jobs:\n', data1['job'].unique())
print('Marital:\n', data1['marital'].unique())
print('Education:\n', data1['education'].unique())
print('Default:\n', data1['default'].unique())
print('Housing:\n', data1['housing'].unique())
print('Loan:\n', data1['loan'].unique())
print('contact:\n', data1['loan'].unique())
print('poutcome:\n', data1['loan'].unique())
print('deposit:\n', data1['loan'].unique())


# default,housing,loan,contact, poutcome,deposit are binary 

# # Exploratory Data Analysis

# In[18]:


#correlation
plt.figure(figsize=(10,6))
sns.heatmap(data1.corr(), cmap = "YlGnBu", annot = True)


# pdays and previous are highly correlated

# In[19]:


#Visualizing the target variable(term deposit)
plt.figure(figsize=(13,7))
data1.groupby('deposit').size().plot(kind='pie', autopct='%.2f',ylabel='percentage')


# In[ ]:





# In[20]:


plt.figure(figsize=(13,7))
sns.histplot(x=data['age'],color='violet',label='Age')


# In[21]:


plt.figure(figsize=(13,7))
sns.countplot(x = 'job', data = data1)


# Management and Blue-collar Job type Clients are maximum in the bank
# There are very less number of housemaid customers in the bank

# In[22]:


plt.figure(figsize=(13,7))
data1.groupby('marital').size().plot(kind='pie', autopct='%.2f',ylabel='percentage')


# Most of the clients in the bank are Married - 56.15% and Single - 32.54%

# In[23]:


plt.figure(figsize=(13,7))
sns.countplot(x = 'education', data = data1)


# Most of the customers in the bank are related to Secondary and Tertiary Category

# In[24]:


plt.figure(figsize=(13,7))
data1.groupby('default').size().plot(kind='pie', autopct='%.2f',ylabel='percentage')


# In[25]:


plt.figure(figsize=(13,7))
data1.groupby('housing').size().plot(kind='pie', autopct='%.2f',ylabel='percentage')


# In[26]:


plt.figure(figsize=(13,7))
data1.groupby('loan').size().plot(kind='pie', autopct='%.2f',ylabel='percentage')


# 98.49% customers in the bank do not have Credit in Default
# 52.46% customers in the bank do not have Housing Loan
# 87.13% customers in the bank do not have Personal Loan
# on comparing with Personal Loan and Housing Loan, Most of the clients subscribed for Housing Loan - 47.54%

# In[27]:


sns.countplot(x = 'poutcome', data = data1)


# From the Outcomes of the previous marketing Campaign most of the results are Unknown. 
# Success rate is very less commpared to failure.
# From the Analysis, on doing Marketing Campaigns there were more Failure than Success.

# In[28]:


import plotly.express as px
fig=px.box(data1,x='deposit',y='age',color='deposit',template='simple_white',color_discrete_sequence=['DeepSkyBlue','LightCoral'],title='<b>Distribution of age based on Term Deposit Status')
fig.update_layout(title_x=0.5,font_family="Times New Roman",legend_title_text="<b>Term Deposit")
fig.show()


# The median age of the clients who subscribed and not subscribed for the term deposit is almost same.

# In[32]:


plt.figure(figsize=(13,7))
sns.boxplot(data=data1, x="deposit", y="balance")
plt.title("Balance vs. Subscription Status")
plt.show()

plt.figure(figsize=(13,7))
sns.boxplot(data=data1, x="deposit", y="duration")
plt.title("Duration vs. Subscription Status")
plt.show()


# In[33]:


# Visualize the relationship between categorical columns and the target variable
sns.set(rc={'figure.figsize':(20,6)})
sns.countplot(x=data1['job'], data=data1, hue=data1['deposit'])
plt.title('Count Plot of job for term deposit')

sns.countplot(data=data1, x="deposit", hue="marital")
plt.title("Counts of Marital Status vs. Subscription Status")
plt.show()

sns.countplot(data=data1, x="deposit", hue="education")
plt.xticks(rotation=45, ha="right")
plt.title("Counts of Education vs. Subscription Status")
plt.show()

sns.countplot(data=data1, x="deposit", hue="default")
plt.title("Counts of Default vs. Subscription Status")
plt.show()

sns.countplot(data=data1, x="deposit", hue="housing")
plt.title("Counts of Housing vs. Subscription Status")

sns.countplot(data=data1, x="deposit", hue="loan")
plt.title("Counts of personal loan vs. Subscription Status")


# In[34]:


sns.boxplot(y=data['pdays'], x=data['deposit'])
plt.title('Box plot of pdays vs term deposit')
plt.xlabel('deposit')


# In[35]:


# Encoding categorical variables
le = LabelEncoder()
data1['job'] = le.fit_transform(data1['job'])
data1['marital'] = le.fit_transform(data1['marital'])
data1['education'] = le.fit_transform(data1['education'])
data1['default'] = le.fit_transform(data1['default'])
data1['housing'] = le.fit_transform(data1['housing'])
data1['loan'] = le.fit_transform(data1['loan'])
data1['contact'] = le.fit_transform(data1['contact'])
data1['month'] = le.fit_transform(data1['month'])
data1['poutcome'] = le.fit_transform(data1['poutcome'])
data1['deposit'] = le.fit_transform(data1['deposit'])


# In[36]:


scaler = StandardScaler()
data1[['age', 'duration', 'campaign', 'pdays', 'previous','balance','day']] = scaler.fit_transform(data1[['age', 'duration', 'campaign', 'pdays', 'previous','balance','day']])


# # CLASSIFICATION MODELS

# In[37]:


X=data1.drop(["deposit",],axis=1)
y=data1["deposit"]


# In[38]:


# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)


# # LOGISTIC REGRESSION

# In[39]:


# Make an instance of the Model
logistic = LogisticRegression()

# Fitting the values for x and y
logistic.fit(x_train,y_train)
logistic.coef_
logistic.intercept_


# In[40]:


# Prediction from test data
prediction = logistic.predict(x_test)
print(prediction)


# In[41]:


# Confusion matrix
cf_matrix = confusion_matrix(y_test, prediction)
print(cf_matrix)
TP = cf_matrix[1,1] # true positive 
TN = cf_matrix[0,0] # true negatives
FP = cf_matrix[0,1] # false positives
FN = cf_matrix[1,0] # false negatives
print('True Positive[TP] =',TP,'\nTrue Negative[TN]=',TN,'\nFalse Positive[FP]=',FP,'\nFalse Negative[FN]=',FN,'\n')


# In[42]:


# Finding Accuracy
logreg = accuracy_score(prediction,y_test)*100
print(logreg)


# In[43]:


# Printing the misclassified values from prediction
print('Misclassified samples: %d' % (y_test != prediction).sum())


# In[44]:


pred_prob_lmod1 = logistic.predict_proba(x_test)[:,1]


# ROC Curve
f, ax = plt.subplots(figsize=(10,6))
fpr, tpr, thresholds = roc_curve(y_test,pred_prob_lmod1)
plt.plot([0, 1], [0, 1], linestyle='--')
# ROC Curve 
plt.plot(fpr, tpr, marker='.',label='AUC = {:.2f}'.format(auc(fpr, tpr)))
ax.set_title('ROC Curve ' +'Logistic'+'\nAUC ROC Score:{0:.4f}'.format(roc_auc_score(y_test,prediction)))
plt.xlabel("False Positive Rate")
plt.ylabel("TruePositive Rate")
plt.show()


# # KNN

# In[45]:


# Storing the K nearest neighbors classifier
KNN = KNeighborsClassifier(n_neighbors = 5)  


# In[46]:


# Fitting the values for X and Y
KNN.fit(x_train, y_train) 


# In[47]:


# Predicting the test values with model
predictionknn = KNN.predict(x_test)


# In[48]:


# Performance metric check
confusion_matrix2 = confusion_matrix(y_test, predictionknn)
print(confusion_matrix2)


# In[49]:


# Calculating the accuracy
knnscore=accuracy_score(y_test, predictionknn)*100
print(knnscore)


# In[50]:


print('Misclassified samples: %d' % (y_test != predictionknn).sum())


# In[51]:


pred_prob_knn = KNN.predict_proba(x_test)[:,1]


# ROC Curve
f, ax = plt.subplots(figsize=(10,6))
fpr, tpr, thresholds = roc_curve(y_test,pred_prob_knn)
plt.plot([0, 1], [0, 1], linestyle='--')
# ROC Curve 
plt.plot(fpr, tpr, marker='.',label='AUC = {:.2f}'.format(auc(fpr, tpr)))
ax.set_title('ROC Curve ' +'KNN'+'\nAUC ROC Score:{0:.4f}'.format(roc_auc_score(y_test,predictionknn)))
plt.xlabel("False Positive Rate")
plt.ylabel("TruePositive Rate")
plt.show()


# # DECISION TREE

# In[52]:


#Fitting the model

dtree = DecisionTreeClassifier()
classf = dtree.fit(x_train,y_train)


# In[53]:


# Applying the model to the x_test

predictiondtree = classf.predict(x_test)
predictiondtree


# In[54]:


# Finding Accuracy

dectree= accuracy_score(predictiondtree,y_test)*100
print(dectree)


# In[55]:


# Confusion Matrix

conf_dtree=confusion_matrix(y_test,predictiondtree)
print(conf_dtree)
TP = conf_dtree[1,1] # true positive 
TN = conf_dtree[0,0] # true negatives
FP = conf_dtree[0,1] # false positives
FN = conf_dtree[1,0] # false negatives
print('True Positive[TP] =',TP,'\nTrue Negative[TN]=',TN,'\nFalse Positive[FP]=',FP,'\nFalse Negative[FN]=',FN,'\n')


# In[56]:


pred_prob_dtree = dtree.predict_proba(x_test)[:,1]


# ROC Curve
f, ax = plt.subplots(figsize=(10,6))
fpr, tpr, thresholds = roc_curve(y_test,pred_prob_dtree)
plt.plot([0, 1], [0, 1], linestyle='--')
# ROC Curve 
plt.plot(fpr, tpr, marker='.',label='AUC = {:.2f}'.format(auc(fpr, tpr)))
ax.set_title('ROC Curve ' +'Dtree'+'\nAUC ROC Score:{0:.4f}'.format(roc_auc_score(y_test,predictiondtree)))
plt.xlabel("False Positive Rate")
plt.ylabel("TruePositive Rate")
plt.show()


# # RANDOM FOREST

# In[57]:


#Fitting the model

rft = RandomForestClassifier(n_estimators=30,criterion='gini',random_state=1,max_depth=10)
rft.fit(x_train, y_train)


# In[58]:


# Applying the model to the x_test

pred_rft= rft.predict(x_test)
pred_rft


# In[59]:


# Finding Accuracy

RanFor = accuracy_score(y_test,pred_rft)*100
print(RanFor)


# In[60]:


# Confusion Matrix

conf_rft=confusion_matrix(y_test,pred_rft)
print(conf_rft)
TP = conf_rft[1,1] # true positive 
TN = conf_rft[0,0] # true negatives
FP = conf_rft[0,1] # false positives
FN = conf_rft[1,0] # false negatives
print('True Positive[TP] =',TP,'\nTrue Negative[TN]=',TN,'\nFalse Positive[FP]=',FP,'\nFalse Negative[FN]=',FN,'\n')


# In[61]:


pred_prob_rft = rft.predict_proba(x_test)[:,1]


# ROC Curve
f, ax = plt.subplots(figsize=(10,6))
fpr, tpr, thresholds = roc_curve(y_test,pred_prob_rft)
plt.plot([0, 1], [0, 1], linestyle='--')
# ROC Curve 
plt.plot(fpr, tpr, marker='.',label='AUC = {:.2f}'.format(auc(fpr, tpr)))
ax.set_title('ROC Curve ' +'Random forest'+'\nAUC ROC Score:{0:.4f}'.format(roc_auc_score(y_test,pred_rft)))
plt.xlabel("False Positive Rate")
plt.ylabel("TruePositive Rate")
plt.show()


# In[62]:


models = pd.DataFrame({'Models': ['Logistic Regression','K-Near Neighbors','Decision Tree Classifier','Random Forest Classifier'],
                       'Score':  [logreg,knnscore,dectree,RanFor]})

models.sort_values(by='Score', ascending=False)


# ## Random forest is the best classification model for this dataset with an accuacy score of 82.92 percentage

# In[ ]:




