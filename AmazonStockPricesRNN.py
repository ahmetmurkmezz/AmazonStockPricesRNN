# -*- coding: utf-8 -*-
import pymongo
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
from sklearn.metrics import mean_squared_error, r2_score


client = MongoClient("mongodb://localhost:27017/")
database = client["local"]
column = database['AmazonStockPrices']

data = pd.DataFrame(list(column.find()))
data = data.drop("_id", axis=1)

data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')

plt.figure(figsize=(12, 6))  
plt.plot(data['Date'],data['Open'], label='Open', color='blue')
plt.plot(data['Date'],data['Close'], label='Close', color='red')
plt.title('Stock Prices Opens')
plt.xlabel('Year')
plt.ylabel('Open')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)

plt.show()



dataOpen = data[['Date','Open']]


length_data = len(dataOpen)
split_ratio = 0.8
length_train = round(length_data * split_ratio)
length_validation = length_data - length_train

print("Data length: ", length_data)
print("Train data length : " , length_train)
print("Validation data length: ", length_validation)

training_data = dataOpen[:length_train].iloc[:,:2]
training_data['Date'] = dataOpen['Date'].dt.date


validation_data = dataOpen[length_train:].iloc[:,:2]
validation_data['Date'] = dataOpen['Date'].dt.date


dataset_train = training_data.Open.values
dataset_train = np.reshape(dataset_train, (-1,1))

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset_train)

x_train = []
y_train = []

steps = 60

for i in range(steps, length_train):
    x_train.append(scaled_data[i-steps:i, 0])
    y_train.append(scaled_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
y_train = np.reshape(y_train, (y_train.shape[0],1))


model = Sequential()
model.add(SimpleRNN(units=128, activation = 'tanh', return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(SimpleRNN(units=64, activation = 'tanh', return_sequences=True))
model.add(Dropout(0.2))
model.add(SimpleRNN(units=64, activation = 'tanh', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mean_squared_error", metrics=['accuracy'])
history = model.fit(x_train,y_train,epochs=50,batch_size=32)



plt.figure(figsize=(10,7))
plt.plot(history.history["loss"])
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.title("Model Loss Değerleri")
plt.show()



plt.figure(figsize=(10,7))
plt.plot(history.history["accuracy"])
plt.xlabel("Epochs")
plt.ylabel("Accuracies")
plt.title("Model Accuracy Değerleri")
plt.show()
    
    
y_pred = model.predict(x_train)
y_pred = scaler.inverse_transform(y_pred)

y_train = scaler.inverse_transform(y_train)


plt.figure(figsize = (30,10))
plt.plot(y_pred, color="blue", label="Predictions")
plt.plot(y_train, color="red", label="Train")
plt.xlabel("Dates")
plt.ylabel("Open Prices")
plt.title("Model")
plt.legend()
plt.show()

validation = validation_data.Open.values
validation = np.reshape(validation,(-1,1))
scaled_validation = scaler.fit_transform(validation)

x_test = []
y_test = []

for i in range(steps,length_validation):
    x_test.append(scaled_validation[i-steps:i, 0])
    y_test.append(scaled_validation[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
y_test = np.reshape(y_test, (-1,1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

print("Prediction: " , predictions[0])
print(r2_score(y_test,predictions))

plt.figure(figsize=(30,10))
plt.plot(predictions, label="Predictions", color="green")
plt.plot(scaler.inverse_transform(y_test), label="Test Datas", color="yellow")
plt.xlabel("Dates")
plt.ylabel("Open Prices")
plt.title("Model Tahmini")
plt.legend()
plt.show()



    
    
    




