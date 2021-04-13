import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score

df = pd.read_csv('admissions_data.csv')
df.drop(["Serial No."],axis = 1,inplace = True)
features = df.iloc[:,:-2]
labels = df.iloc[:,-1]
labels = labels.to_frame()

X_train, X_test, y_train, y_test = train_test_split(
features, labels, test_size = 0.8)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(X_train)
features_test_scaled = scaler.transform(X_test)

model = Sequential()
model.add(layers.Input(shape=(features_scaled.shape[1])))
model.add(layers.Dense(32,activation = 'relu'))
model.add(layers.Dense(32,activation = 'relu'))
model.add(layers.Dense(1))
print(model.summary())
model.compile(loss = 'mse' , optimizer = 'adam', metrics = ['mae'])
model.fit(features_scaled,y_train,epochs = 20,batch_size = 40)
model.evaluate(features_test_scaled,y_test,batch_size = 30,
 verbose = 1
)
