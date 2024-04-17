#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


match=pd.read_csv(r"C:\Users\ABI PRIYANKA\Downloads\IPL Matches 2008-2020.csv")


# In[4]:


match.head()


# In[5]:


match_data = match.loc[(match.team1== "Chennai Super Kings") | (match.team2 == "Chennai Super Kings")]


# In[6]:


match_data.info()


# In[7]:


# Convert 'date' column to a usable format
match_data['date'] = pd.to_datetime(match['date'])


# In[8]:


# Extract the year from the 'date' column
match_data['season'] = match_data['date'].dt.year

# Count the matches played in each season
matches_per_season = match_data['season'].value_counts()

print("Matches played in each season:")
print(matches_per_season)


# In[9]:


len(np.unique(match_data['id']))


# In[10]:


match_data.info()


# In[11]:


match_data['winner'].value_counts()


# In[12]:


cities = match_data['city'].value_counts().head(10)
plt.figure(figsize=(12,12))
c1 = sns.countplot(y= 'city',order=match_data['city'].value_counts().iloc[:10].index,data = match_data,palette = 'mako')
c1.bar_label(c1.containers[0],size = 15)
plt.ylabel('Cities',fontsize=12)
plt.xlabel('No: of matches',fontsize=12)
plt.title('Top 10 Cities in which matches where held ',fontsize=15)
plt.show()


# In[13]:


pom = match_data['player_of_match'].value_counts().iloc[:10]
plt.figure(figsize=(12,12))
c1 = sns.countplot(y= 'player_of_match',order=pom.index,data = match_data,palette = 'mako')
c1.bar_label(c1.containers[0],size = 15)
plt.ylabel('Player',fontsize=12)
plt.xlabel('No: of matches',fontsize=12)
plt.title('Top 10 Players who received POM',fontsize=15)
plt.show()


# In[14]:


venues = match_data['venue'].value_counts().head(10)
plt.figure(figsize=(12,12))
c1 = sns.countplot(y='venue',order=venues.index,data = match_data,palette = 'mako')
c1.bar_label(c1.containers[0],size = 15)
plt.ylabel('Venues',fontsize=12)
plt.xlabel('No: of matches',fontsize=12)
plt.title('Top 10 Venues where matches were held',fontsize=15)
plt.show()


# In[15]:


match_app=pd.concat([match_data['team1'],match_data['team2']])
match_app=match_app.value_counts().reset_index()
match_app.columns=['Team','Total Matches']
match_app


# In[16]:


toss_wins = match_data['toss_winner'].value_counts()
plt.figure(figsize=(12,12))
c1 = sns.countplot(y='toss_winner',order=toss_wins.index,data = match_data,palette = 'mako')
c1.bar_label(c1.containers[0],size = 15)
plt.ylabel('Venues',fontsize=12)
plt.xlabel('No: of tosses won',fontsize=12)
plt.title('Teams ranked on basis of winning toss',fontsize=15)
plt.show()


# In[17]:


# Filter data for CSK matches where CSK won the toss and won the match
csk_won_toss_won_match = match_data[(match_data['toss_winner'] == 'Chennai Super Kings') & (match_data['winner'] == 'Chennai Super Kings')]

# Filter data for CSK matches where CSK won the toss but lost the match
csk_won_toss_lost_match = match_data[(match_data['toss_winner'] == 'Chennai Super Kings') & (match_data['winner'] != 'Chennai Super Kings')]

# Create subplots to display both countplots side by side
plt.figure(figsize=(12, 6))

# Plot for CSK winning the toss and winning the match
plt.subplot(1, 2, 1)
sns.countplot(x='winner', data=csk_won_toss_won_match)
plt.title('CSK Winning the Toss and Winning the Match')
plt.xlabel('Winner')
plt.ylabel('Count')

# Count the instances of CSK winning the toss and winning the match
csk_won_match_count = csk_won_toss_won_match.shape[0]
print("Count of CSK winning the toss and winning the match:", csk_won_match_count)

# Plot for CSK winning the toss and losing the match
plt.subplot(1, 2, 2)
sns.countplot(x='winner', data=csk_won_toss_lost_match)
plt.title('CSK Winning the Toss and Losing the Match')
plt.xlabel('Winner')
plt.ylabel('Count')

# Count the instances of CSK winning the toss but losing the match
csk_lost_match_count = csk_won_toss_lost_match.shape[0]
print("Count of CSK winning the toss but losing the match:", csk_lost_match_count)

# Adjust layout and display the plots
plt.tight_layout()
plt.show()


# In[18]:


##Classification algorithms


# In[22]:


# Encoding categorical variables using one-hot encoding
match_encoded = pd.get_dummies(match_data, columns=['city', 'team1', 'team2', 'toss_winner', 'toss_decision', 'venue'])


# In[24]:


match_encoded


# In[32]:


# Create a binary encoding for the target variable 'y'
y = match_encoded['winner'].apply(lambda x: 1 if x == 'Chennai Super Kings' else 0)

# Drop the 'winner' column from the feature matrix 'X'
X = match_encoded.drop(columns=['id', 'date', 'player_of_match', 'winner', 'method', 'umpire1','umpire2'])

# Convert categorical columns to numeric using one-hot encoding
X = pd.get_dummies(X)

# Handle missing values (if any)
X.fillna(0, inplace=True)  # Replace missing values with 0, assuming 0 is not a valid value in the dataset

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier with the best hyperparameters from the previous step
rf_classifier = RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=42)

# Train the model using all features
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy and print classification report
accuracy_all_features = accuracy_score(y_test, y_pred)
print("Accuracy with all features:", accuracy_all_features)

print("Classification Report with all features:")
print(classification_report(y_test, y_pred))

# Find feature importances
feature_importances = rf_classifier.feature_importances_

# Create a DataFrame to show feature importances
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(importance_df)

# Drop the least impactful variable
least_impactful_variable = importance_df.iloc[-1]['Feature']
X_train_reduced = X_train.drop(columns=[least_impactful_variable])
X_test_reduced = X_test.drop(columns=[least_impactful_variable])

# Retrain the model without the least impactful variable
rf_classifier_reduced = RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=42)
rf_classifier_reduced.fit(X_train_reduced, y_train)

# Predict on the test set using the reduced feature set
y_pred_reduced = rf_classifier_reduced.predict(X_test_reduced)

# Calculate accuracy and print classification report with reduced features
accuracy_reduced_features = accuracy_score(y_test, y_pred_reduced)
print("Accuracy with reduced features:", accuracy_reduced_features)

print("Classification Report with reduced features:")
print(classification_report(y_test, y_pred_reduced))


# In[34]:


# Initialize the Decision Tree Classifier with the best hyperparameters (you can tune these as well)
dt_classifier = DecisionTreeClassifier(max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=42)

# Train the model using all features
dt_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = dt_classifier.predict(X_test)

# Calculate accuracy and print classification report
accuracy_all_features = accuracy_score(y_test, y_pred)
print("Accuracy with all features:", accuracy_all_features)

print("Classification Report with all features:")
print(classification_report(y_test, y_pred))

# Find feature importances (not available in DecisionTreeClassifier, but you can still get the most important feature)
most_important_variable = X_train.columns[dt_classifier.feature_importances_.argmax()]
print("Most important variable:", most_important_variable)

# Drop the most impactful variable (in this case, we are dropping the most important variable for illustration)
X_train_reduced = X_train.drop(columns=[most_important_variable])
X_test_reduced = X_test.drop(columns=[most_important_variable])

# Retrain the model without the most impactful variable
dt_classifier_reduced = DecisionTreeClassifier(max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=42)
dt_classifier_reduced.fit(X_train_reduced, y_train)

# Predict on the test set using the reduced feature set
y_pred_reduced = dt_classifier_reduced.predict(X_test_reduced)

# Calculate accuracy and print classification report with reduced features
accuracy_reduced_features = accuracy_score(y_test, y_pred_reduced)
print("Accuracy with reduced features:", accuracy_reduced_features)

print("Classification Report with reduced features:")
print(classification_report(y_test, y_pred_reduced))


# In[35]:


# Initialize the Logistic Regression model
logreg_classifier = LogisticRegression(random_state=42)

# Train the model using all features
logreg_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = logreg_classifier.predict(X_test)

# Calculate accuracy and print classification report
accuracy_all_features = accuracy_score(y_test, y_pred)
print("Accuracy with all features:", accuracy_all_features)

print("Classification Report with all features:")
print(classification_report(y_test, y_pred))

# Find feature importances (not available in Logistic Regression)
# We can print the coefficients as a measure of feature importance
coefficients = logreg_classifier.coef_[0]
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': coefficients})
importance_df = importance_df.sort_values(by='Coefficient', ascending=False)

print("Feature Importances:")
print(importance_df)

# Drop the least impactful variable (the one with the smallest absolute coefficient)
least_impactful_variable = importance_df.iloc[-1]['Feature']
X_train_reduced = X_train.drop(columns=[least_impactful_variable])
X_test_reduced = X_test.drop(columns=[least_impactful_variable])

# Retrain the model without the least impactful variable
logreg_classifier_reduced = LogisticRegression(random_state=42)
logreg_classifier_reduced.fit(X_train_reduced, y_train)

# Predict on the test set using the reduced feature set
y_pred_reduced = logreg_classifier_reduced.predict(X_test_reduced)

# Calculate accuracy and print classification report with reduced features
accuracy_reduced_features = accuracy_score(y_test, y_pred_reduced)
print("Accuracy with reduced features:", accuracy_reduced_features)

print("Classification Report with reduced features:")
print(classification_report(y_test, y_pred_reduced))


# In[40]:


# Initialize the KNN classifier
knn_classifier = KNeighborsClassifier()

# Train the model using all features
knn_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = knn_classifier.predict(X_test)

# Calculate accuracy and print classification report
accuracy_all_features = accuracy_score(y_test, y_pred)
print("Accuracy with all features:", accuracy_all_features)

print("Classification Report with all features:")
print(classification_report(y_test, y_pred))


# In[ ]:




