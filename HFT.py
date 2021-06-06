#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[3]:


series = pd.read_csv('gaz1.csv', ';', header=0, parse_dates=[0], index_col=['<DATE>'], squeeze=True)
series = series.drop(["<TICKER>","<PER>","<TIME>","<VOL>"], 1)
series.plot(figsize=(20,10))


# In[ ]:


# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(series.LAST); axes[0, 0].set_title('Original Series')
plot_acf(series.LAST, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(series.LAST.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(series.LAST.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(series.LAST.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(series.LAST.diff().diff().dropna(), ax=axes[2, 1])

plt.show()


# In[24]:


model = ARIMA(series, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())


# In[4]:


X = series.LAST
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
print(len(test))
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()


# In[ ]:




