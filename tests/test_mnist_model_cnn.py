import sys
sys.path.append('/home/amishra/appspace/Do-It-Yourslef-data-science/')
import  predictive_modeling_part1.mnist_model_only_fc as m

from common.h5_reader_writer import  H5Reader
import  predictive_modeling_part1.mnist_cnn as cnn


import numpy

import tensorflow

import matplotlib.pyplot as plt

from PIL import Image
import matplotlib.pyplot as plt
plt.ion()
def train_mnist_cnn_model():
    config = {

        'learning_rate': 0.0001,
        'num_epochs': 100,
        'print_frequency': 10,
    }

    x = m.mnist.train.images
    y = m.mnist.train.labels
    lr = cnn.MnistModel(config)
    learned_params = lr.train(train_X=x, train_Y=y)
    lr.write_params_to_file("../trained_model/mnist_cnn_model_3_layers_adam.h5", learned_params)


def test_mnist_cnn_model():

    config, learned_params = H5Reader.read("../trained_model/mnist_cnn_model_3_layers_adam.h5")
    lr = cnn.MnistModel(config)

    x_test = m.mnist.test.images
    y_test = m.mnist.test.labels

    y_hat = lr.predict(learned_params, X = x_test)
    y_hat = y_hat[0]
    print(y_hat[0])

    for i in range(900, 1000):
        im1 = x_test[i,:]
        im1 = im1.reshape(28,28)
        plt.imshow(im1)
        y_true = numpy.argmax(y_test[i,:])
        plt.suptitle('The image is classified as {} but the true image is {}'.format(numpy.argmax(y_hat[i, :]), y_true))
        plt.show()
        plt.waitforbuttonpress()





def train_mnist_model():
    config = {

        'learning_rate': 0.08,
        'num_epochs': 10,
        'print_frequency': 100,
    }

    x = m.mnist.train.images
    y = m.mnist.train.labels
    lr = m.MnistModel(config)
    learned_params = lr.train(train_X=x, train_Y=y)
    lr.write_params_to_file("../trained_model/mnist_model_3_layers_adam.h5", learned_params)


def test_mnist_model():

    config, learned_params = H5Reader.read("../trained_model/mnist_model_3_layers.h5")
    lr = m.MnistModel(config)

    x_test = m.mnist.test.images
    y_test = m.mnist.test.labels

    y_hat = lr.predict(learned_params, X=x_test)
    y_hat = y_hat[0]
    print(y_hat.shape)


    for i in range(900, 1000):
        im1 = x_test[i, :]
        im1 = im1.reshape(28,28)
        plt.imshow(im1)
        y_true = numpy.argmax(y_test[i, :])

        plt.suptitle('The image is classified as {} but the true image is {}'.format(numpy.argmax(y_hat[i, :]), y_true))
        plt.show()
        plt.waitforbuttonpress()


if __name__=="__main__":
    test_mnist_cnn_model()


