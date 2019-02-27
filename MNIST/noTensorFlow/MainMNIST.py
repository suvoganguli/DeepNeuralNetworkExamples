import pickle
import gzip, numpy
import matplotlib.pyplot as plt
import numpy as np

def get_data():

    # Load the dataset
    with gzip.open('mnist.pkl.gz','rb') as ff :
        u = pickle._Unpickler(ff)
        u.encoding = 'latin1'
        train, val, test = u.load()

    print("Training set = ", str(train[0].shape))
    print("Training label = ", str(train[1].shape))
    print("Validation set = ", str(val[0].shape))
    print("Validation label = ", str(val[1].shape))
    print("Test set = ", str(test[0].shape))
    print("Test label = ", str(test[1].shape))

    if False:
        img_array = train[0][0]
        label_array = train[1][0]
        print(label_array)

        img = img_array.reshape((28,28))
        plt.imshow(img,cmap='gray')
        plt.show()

    return train, val, test


def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],100)
        self.weights2   = np.random.rand(100,10)
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1

        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

train, val, test = get_data()

X_all = train[0]
y_all = train[1]

n_subset = 500
X = X_all[:n_subset,:]
y = y_all[:n_subset]

nb_classes = 10
data = [[y]]
y_onehotencode = indices_to_one_hot(data, nb_classes)

if True:
    nn = NeuralNetwork(X,y_onehotencode)
    for i in range(10):
        nn.feedforward()
        nn.backprop()

    print(nn.output)