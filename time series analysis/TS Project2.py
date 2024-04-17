#!/usr/bin/env python
# coding: utf-8

# 1) Apply various frequencies to resample the given time series data

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[2]:


# Read the shampoo sales data from the text file
data = pd.read_csv(r"C:\Users\ABI PRIYANKA\Downloads\shampoo (1).txt", delimiter=',')


# In[3]:


data


# In[4]:


data['Year'] = data['Month'].apply(lambda x: int(x.split('-')[0]) + 2000)
data['Month'] = data['Month'].apply(lambda x: int(x.split('-')[1]))
data['Date'] = data.apply(lambda x: pd.Timestamp(year=int(x['Year']), month=int(x['Month']), day=1), axis=1)
data.set_index('Date', inplace=True)
data.drop(['Year', 'Month'], axis=1, inplace=True)
data['Sales'] = data['Sales'].astype(float)


# In[5]:


data.head()


# In[6]:


data.describe()


# 1)Apply various frequencies to resample the given time series data

# In[11]:


resampled_daily = data.resample('D').asfreq()
resampled_monthly = data.resample('M').sum() # Monthly frequency
resampled_quarterly = data.resample('Q').sum()# Quarterly frequency
resampled_data = data.resample('A').sum()  # Annual frequency
resampled_weekly = data.resample('W').mean()  # Resample to weekly frequency


# In[12]:


print("Daily Resampled Data:")
print(resampled_daily.head())
print()

print("Weekly Resampled Data:")
print(resampled_weekly.head())
print()

print("Monthly Resampled Data:")
print(resampled_monthly.head())
print()

print("Quarterly Resampled Data:")
print(resampled_quarterly.head())
print()

print("Annually Resampled Data:")
print(resampled_data.head())


# 2)Calculate moving average with appropriate rolling window to analyze the data points
# by creating a series of averages of different subsets of the full data set

# In[13]:


# Calculate the moving average with a rolling window of 3
rolling_mean = data['Sales'].rolling(window=3).mean()


# In[14]:


# Print the moving averages
print("Moving Averages:")
print(rolling_mean)


# 3)Plot the given time series data to observe the trend and seasonality of the signal 
# and comment your inferences

# In[15]:


# Plot the sales data
plt.plot(data.index, data['Sales'])
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Shampoo Sales Data')
plt.show()


# 4)Decompose the sales time series data into multiple components based on 
# appropriate period

# In[16]:


import statsmodels.api as sm

# Decompose the sales data
decomposition = sm.tsa.seasonal_decompose(data['Sales'], model='additive')

# Plot the decomposed components
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
decomposition.observed.plot(ax=ax1)
ax1.set_ylabel('Observed')
decomposition.trend.plot(ax=ax2)
ax2.set_ylabel('Trend')
decomposition.seasonal.plot(ax=ax3)
ax3.set_ylabel('Seasonality')
decomposition.resid.plot(ax=ax4)
ax4.set_ylabel('Residuals')
plt.tight_layout()
plt.show()


# 5)Check if the given time series is stationary or not
# 

# In[18]:


#Ho: The data is non-stationary
##H1: The data is stationary

adf_test=adfuller(data['Sales'],autolag='AIC')
print("1. ADF test:",adf_test[0])
print("2. ADF test p-value:",adf_test[1])
print("3. Number of lags :", adf_test[2])


# The null hypothesis of the ADF test is that time series is non stationary.
# The output shows that p-value is 1.0 which is greater than significance level (0.05). 
# Hence, we failed to reject the null hypothesis and conclude that the series is non stationary.

# 6)Visualize auto-correlation function (ACF), partial auto-correlation function (PACF) 
# and comment your inference

# In[27]:


plot_acf(data['Sales'].iloc[13:],lags=10)
plot_pacf(data['Sales'].iloc[13:],lags=10)
plt.show()


# From the plot of seasonal difference, we can see that PACF cuts-off after lag3 which indicates that ACF slowly decays

# In[ ]:




