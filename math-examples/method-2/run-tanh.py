__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import os
import json
import time
import math
import matplotlib.pyplot as plt
from core.data_processor_tanh import DataLoader
from core.model_tanh import Model
import numpy as np


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.pause(10)
    #plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
	# Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.pause(3)
    #plt.show()


def main():
    configs = json.load(open('config-tanh.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    model = Model()
    model.build_model(configs)
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    # in-memory training
    model.train(
	x,
	y,
	epochs = configs['training']['epochs'],
	batch_size = configs['training']['batch_size'],
        save_dir=configs['model']['save_dir']
    )

    '''
    # out-of memory generative training
    print(data.len_train)
    print(configs['data']['sequence_length'])
    steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    print('n = ', steps_per_epoch)
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
            normalise=configs['data']['normalise']
        ),
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_dir=configs['model']['save_dir']
    )
    '''

    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    # predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
    # predictions = model.predict_sequence_full(x_test[0:], configs['data']['sequence_length'])
    # predictions = model.predict_point_by_point(x_test)

    # plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
    # plot_results(predictions, y_test)

    # ---------------------------

    y_pred = []
    plt.figure(10) 
    plt.plot(y_test, 'b')

    seq_len=configs['data']['sequence_length']
    for k in np.arange(0,len(y_test)-seq_len,10):

       x_pred = x_test[k]
       
       y_pred = []
       for j in range(seq_len):
           y_pred_j = model.predict_sequence_full([x_pred], configs['data']['sequence_length'])
           y_pred.append(y_pred_j)

           x_pred = x_test[k+j+1]
           x_pred[-1] = y_pred_j[0] 

        
       plt.plot(np.arange(k,k+seq_len),y_pred, 'r')
       plt.pause(0.01)






if __name__ == '__main__':
    main()
