#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from keras.models import load_model



dataset = pd.read_csv('DATA\gaz1.csv',';', index_col=['<DATE>'])

values = dataset.LAST
values.plot(figsize=(20,10))

values_train = values[:int(len(values)*0.8)]
values_test = values[int(len(values)*0.8):]

scaler = MinMaxScaler(feature_range=(0,1))

new_data = scaler.fit_transform(dataset["LAST"].values.reshape(-1,1))

pred_ticks = 100

x_train = []
y_train = []

for i in range(1, pred_ticks):
    arr = []
    for y in range(pred_ticks - i):
        arr.append(new_data[0,0])
    for y in new_data[:i,0]:
        arr.append(y)
    x_train.append(arr)
    y_train.append(new_data[i,0])

for x in range(pred_ticks, len(new_data)):
    x_train.append(new_data[x - pred_ticks:x, 0])
    y_train.append(new_data[x,0])
    
x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 25, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 5))
model.add(Dropout(0.2))
model.add(Dense(units = 1))

model.compile(optimizer = "adam", loss = "mean_squared_error")



model.fit(x_train, y_train, epochs = 10, batch_size = 1000)


price = model.predict(x_train)
price = scaler.inverse_transform(price)
plt.plot(price, color = "black")


dataset2 = pd.read_csv('DATA\gaz5.csv',';', index_col=['<DATE>'])

values2 = dataset2.LAST

scaler = MinMaxScaler(feature_range=(0,1))
x_test = []
new_data2 = scaler.fit_transform(dataset2["LAST"].values.reshape(-1,1))

pred_ticks = 25
x_test = []


for i in range(1, pred_ticks):
    arr = []
    for y in range(pred_ticks - i):
        arr.append([new_data2[0,0]])
    for y in new_data2[:i,0]:
        arr.append([y])
    x_test.append(arr)

for x in range(pred_ticks, len(new_data2)):
    arr = []
    for i in new_data2[x - pred_ticks:x, 0]:
        arr.append([i])
    x_test.append(arr)

x_test = np.array(x_test)

price2 = model.predict(x_test)
price2 = scaler.inverse_transform(price2)

plt.plot(price2, color = "black")
values2.plot(figsize=(20,10))

values2Array  = []
for i in values2:
    values2Array.append([i])
values2Array = np.array(values2Array)

flag = 0
iprev = 0
jprev = 0
profit = 1000
for i,j in zip(values2Array, price2):
    if flag:
        if (i[0] >= iprev and j[0] >= jprev) or (i[0] <= iprev and j[0] <= jprev):
            profit+=1
        else:
            profit-=1
    else:
        flag = 1
    iprev = i
    jprev = j
print(profit)


flag = 0
iprev = 0
jprev = 0
percentArrayReal = []
percentArrayPredicted = []
for i,j in zip(values2Array, price2):
    if flag:
        percentArrayReal.append(iprev/i - 1)
        percentArrayPredicted.append(jprev/j - 1)
    else:
        flag = 1
    iprev = i
    jprev = j


Min = np.min(percentArrayPredicted)
Mean = np.mean(percentArrayPredicted)
Max = np.max(percentArrayPredicted)
arrayBalance = [Min, (Mean + Min)/2, Mean, (Mean + Max)/2, Max]

flag = 0
iprev = 0
jprev = 0
balance = 0
counterOfShares = 0

for i,j in zip(values2Array, price2):
    if flag:
        Difference = jprev/j - 1
        if arrayBalance[0] < Difference < arrayBalance[1]:
            balance -= 4*i
            counterOfShares+=4
        elif arrayBalance[1] < Difference < arrayBalance[2]:
            balance -= 2*i
            counterOfShares+=2
        elif arrayBalance[2] < Difference < arrayBalance[3]:
            balance += 2*i
            counterOfShares-=2
        else:
            balance += 4*i
            counterOfShares-=4
    else:
        flag = 1
    iprev = i
    jprev = j
print(counterOfShares*values2Array[-1] + balance)



