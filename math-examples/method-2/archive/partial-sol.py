import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential

def normalize_windows(window_data, single_window=False):
	'''Normalise window with a base value of zero'''
	normalised_data = []
	window_data = [window_data] if single_window else window_data
	for window in window_data:
	    normalised_window = []
	    for col_i in range(window.shape[1]):
                normalized_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalized_window.append(normalised_col)
            # reshape and transpose array back into original multidimensional format
	    normalized_window = np.array(normalized_window).T 
	    normalized_data.append(normalized_window)
	return np.array(normalised_data)



def next_window(data_train, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = data_train[i:i+seq_len]
        window = normalize_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

# ----------------------------------------------------------

filename = 'data/sinewave.csv'

# get data
dataframe = pd.read_csv(filename)

split = 0.8
normalize = False

i_split = int(len(dataframe) * split)
cols = ['sinewave']
data_train = dataframe.get(cols).values[:i_split]
data_test  = dataframe.get(cols).values[i_split:]
len_train = len(data_train)
len_test = len(data_test)

seq_len = 50

# get data windows
data_x = []
data_y = []
for i in range(len_train - seq_len):

    window = data_train[i:i+seq_len]
    x, y = next_window(data_train, i, seq_len, normalize)

    data_x.append(x)
    data_y.append(y)

x = np.array(data_x)
y = np.array(data_y)


# build model
input_timesteps = 49
input_dim = 1

model = Sequential()
model.add(LSTM(50, input_shape=(input_timesteps, input_dim), return_sequences=True))
model.add(Dropout(0.05))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.05))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')





