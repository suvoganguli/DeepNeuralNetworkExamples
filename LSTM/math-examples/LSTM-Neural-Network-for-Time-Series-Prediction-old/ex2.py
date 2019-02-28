from pandas import read_csv
import math
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential

filename = "data/sinewave.csv"

dataframe = read_csv(filename, header=0, index_col=0, squeeze=True)
split = 0.8
i_split = int(len(dataframe) * split)
data_train = dataframe.get(cols).values[:i_split]
data_test  = dataframe.get(cols).values[i_split:]

input_timesteps = 25
input_dim = 1

model = Sequential()
model.add(LSTM(50, input_shape=(input_timesteps, input_dim), return_sequences=True))
model.add(Dropout(0.05))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.05))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
