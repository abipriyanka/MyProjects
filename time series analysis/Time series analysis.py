#!/usr/bin/env python
# coding: utf-8

# Explore the given data and comment key inferences

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import lag_plot
from math import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA


import warnings
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv(r"C:\Users\ABI PRIYANKA\Downloads\perrin-freres-monthly-champagne.csv",parse_dates=True,index_col=0)
data.head()


# In[3]:


data.info()


# In[4]:


data.describe()


# In[5]:


data['Year']=data.index.year
data['Months']=data.index.month_name()


# In[6]:


data.head()


# In[7]:


plt.figure(figsize=(8,5))
data.groupby('Year')['Sales'].mean().plot(kind='bar')
plt.show()


# 1)HIGHEST SALES DURING 1969 FOLLOWED BY 1971
# 2)ALL THE YEARS HAVE PRODUCED GOOD SALES RECORD

# In[8]:


plt.figure(figsize=(8,5))
data.groupby('Months')['Sales'].mean().plot(kind='bar')
plt.show()


# In[9]:


plt.figure(figsize=(8,5))
data.groupby('Months')['Sales'].mean().reindex(index=['January','February','March','April','May','June','July','August','September','October','November','December']).plot(kind='bar')
plt.show()


# HIGHEST SALES RECORDED DURING THE MONTH OF DECEMBER. NOVEMBER HAS A SALES RECORD CROSSING 8000. AUGUST HAS THE LEAST SALES 

# In[10]:


month=pd.pivot_table(data,values='Sales',index='Months',columns='Year')
month=month.reindex(index=['January','February','March','April','May','June','July','August','September','October','November','December'])


# In[11]:


month


# In[12]:


month.plot(figsize=(8,6))
plt.show()


# ALL THE YEARS FACED A DOWNFALL IN SALES DURING THE MONTH OF AUGUST

# In[13]:


data.plot()


# In[14]:


seaonal=seasonal_decompose(data['Sales'],model='multiplicative')
seaonal.plot()
plt.show()


# SALES OVER THE YEARS HAS AN INCREASING TREND BUT IT SHOWS SEASONALITY

# Check if the given data is stationary or non-stationary
# 

# In[15]:


#Ho: The data is non-stationary
##H1: The data is stationary

adf_test=adfuller(data['Sales'],autolag='AIC')
print("1. ADF test:",adf_test[0])
print("2. ADF test p-value:",adf_test[1])
print("3. Number of lags :", adf_test[2])


# The null hypothesis of the ADF test is that time series is non-stationary. The output shows that p-value is (0.3691) which is greater than significance level (0.05). Hence, we failed to reject the null hypothesis and conclude that the series is non stationary.

# In[ ]:


#data['Sales_diff_1'] = data['Sales'] - data['Sales'].shift(1)
#data.head()


# In[16]:


# Perform seasonal differencing
data['seasonal_differenced_data'] = data['Sales'].diff(12).dropna()


# In[17]:


adf_test=adfuller(data['seasonal_differenced_data'].dropna(),autolag='AIC')
print("1. ADF test:",adf_test[0])
print("2. ADF test p-value:",adf_test[1])
print("3. Number of lags :", adf_test[2])


# After doing 12 month-seasonal difference we find that the output shows that p-value is (2.0605e^-11) which is lesser than significance level (0.05). Hence, we reject the null hypothesis and conclude that the series is stationary.

# Define the order of ARIMA model using ACF and PACF plots

# In[19]:


plot_acf(data['seasonal_differenced_data'].iloc[13:],lags=30)
plot_pacf(data['seasonal_differenced_data'].iloc[13:],lags=30)
plt.show()


# The Autoregressive (AR(p)) value is 1. The Moving Average (MA(q) is 1. We can build our ARIMA model with parameters (1,1,1).

# Build ARIMA and Seasonal ARIMA models to predict the sales

# In[20]:


from statsmodels.tsa.arima.model import ARIMA

# Build ARIMA model
model = ARIMA(data['Sales'], order=(1, 1, 1))
arima_model = model.fit()



# In[21]:


arima_model.summary()


# In[23]:


data.shape


# In[24]:


data['pred']=arima_model.predict(start=85,end=105,dynamic=True)


# In[25]:


data[['Sales','pred']].plot(figsize=(10,8))


# From the plot we can see that our forecast is not efficient. This is because ARIMA model not consider the seasonal part in model building. We can try with Seasonal Arima model.

# In[26]:


#seasonal ARIMA
model2=sm.tsa.statespace.SARIMAX(data['Sales'],order=(1,1,1),seasonal_order=(1,1,1,12)).fit()
model2.summary()


# In[28]:


data['forecast']=model2.predict(start=85,end=105,dynamic=True)
data[['Sales','forecast']].plot(figsize=(10,7))


# From the above plot, the forecast value fitted almost accurately on the actual value

# In[ ]:




