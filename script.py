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
print(features.columns)
print(labels.columns)
skalowanie = StandardScaler()
df_scaled = skalowanie.fit_transform(df)
