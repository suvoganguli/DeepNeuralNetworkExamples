import simplejson as json
import os
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM

class DataLoader():

	def __init__(self, filename, split, cols):
		dataframe = pd.read_csv(filename)
		i_split = int(len(dataframe) * split)
		#print(dataframe)
		self.data_train = dataframe.values[:i_split]
		self.data_test  = dataframe.values[i_split:]
		self.len_train  = len(self.data_train)
		self.len_test   = len(self.data_test)
		self.len_train_windows = None

	def get_train_data(self, seq_len, normalise):
		data_x = []
		data_y = []
		for i in range(self.len_train - seq_len):
			x, y = self._next_window(i, seq_len, normalise)
			data_x.append(x)
			data_y.append(y)
		return np.array(data_x), np.array(data_y)


class Model():

	def __init__(self):
		self.model = Sequential()

	def build_model(self, configs):
		#timer = Timer()
		#timer.start()

		for layer in configs['model']['layers']:
			neurons = layer['neurons'] if 'neurons' in layer else None
			dropout_rate = layer['rate'] if 'rate' in layer else None
			activation = layer['activation'] if 'activation' in layer else None
			return_seq = layer['return_seq'] if 'return_seq' in layer else None
			input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
			input_dim = layer['input_dim'] if 'input_dim' in layer else None

			if layer['type'] == 'dense':
				self.model.add(Dense(neurons, activation=activation))
			if layer['type'] == 'lstm':
				self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
			if layer['type'] == 'dropout':
				self.model.add(Dropout(dropout_rate))

		self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

		print('[Model] Model Compiled')
		#timer.stop()

configs = json.load(open('config.json', 'r'))

data = DataLoader(
	os.path.join('data', configs['data']['filename']),
	configs['data']['train_test_split'],
	configs['data']['columns']
)

model = Model()
model.build_model(configs)
x, y = data.get_train_data(
	seq_len = configs['data']['sequence_length'],
	normalise = configs['data']['normalise']
)

model.train(
	x,
	y,
	epochs = configs['training']['epochs'],
	batch_size = configs['training']['batch_size']
)

x_test, y_test = data.get_test_data(
	seq_len = configs['data']['sequence_length'],
	normalise = configs['data']['normalise']
)




