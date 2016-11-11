
from predictive_modelling_part_0.deep_poly_fit import DeepPolyFit
import numpy
from matplotlib import pyplot as plt
import csv




def test_deep_first_order_polynomial():
    config = {
        'num_hidden_layers': 3,
        'order_poly':3,
        'learning_rate': 0.01,
        'num_epochs': 5000,
        'print_frequency': 1000,

        'hidden_layers': [
            {'op_name': 'h1', 'var_name': 'w1', 'shape': [-1, 35]},
            {'op_name': 'h2', 'var_name': 'w2', 'shape': [-1, 15]},
            {'op_name': 'h3', 'var_name': 'w3', 'shape': [-1, 7]},
            {'op_name': 'h4', 'var_name': 'w4', 'shape': [-1, 17]},
            {'op_name': 'y_hat', 'var_name': 'w5', 'shape': [-1, 1]}
        ]

    }
    lr = DeepPolyFit(config)
    train_x = numpy.linspace(-1,1,1000, dtype=numpy.float32)
    print(train_x)
    train_x = train_x.reshape(-1,1)
    train_y = 5 + 2*train_x + 3*train_x*train_x
    learned_params = lr.train(train_X = train_x, train_Y = train_y)
    lr.update_tensors_with_learned_params(learned_params)
    y_hat = lr.predict(learned_params, X = train_x)
    y_hat = numpy.asarray(y_hat, dtype = numpy.float32, order = None)
    y_hat = y_hat.reshape(-1,1)
    plt.plot(train_x, train_y, 'ro')
    plt.hold(True)
    plt.plot(train_x, y_hat)
    plt.show()
    #error = lr.error(learned_params, X=train_x, Y=train_y)
   # print("error", error)

def test_deep_sinusoidal_regression():
    config = {
        'num_hidden_layers': 4,
        'order_poly': 4,
        'learning_rate': 0.01,
        'num_epochs': 10000,
        'print_frequency': 100,

        'hidden_layers': [
            {'op_name': 'h1', 'var_name': 'w1', 'shape': [-1, 35]},
            {'op_name': 'h2', 'var_name': 'w2', 'shape': [-1, 15]},
            {'op_name': 'h3', 'var_name': 'w3', 'shape': [-1, 7]},
            {'op_name': 'h4', 'var_name': 'w4', 'shape': [-1, 17]},
            {'op_name': 'y_hat', 'var_name': 'w5', 'shape': [-1, 1]}
        ]

    }

    train_x = numpy.linspace(-3,3, 10000, dtype = numpy.float32)
    train_x = train_x.reshape(-1,1)
    train_y = numpy.sin(train_x)
    predictor = DeepPolyFit(config, train_x, train_y)
    #learned_params =  lr.train(train_X = train_x, train_Y = train_y, filename='/tmp/sin.h5')
    #lr.update_tensors_with_learned_params(learned_params)
    y_hat = predictor.predict(X=train_x)
    y_hat = numpy.asarray(y_hat, dtype=numpy.float32, order=None)
    y_hat = y_hat.reshape(-1, 1)
    plt.plot(train_x, train_y, 'r')
    plt.hold(True)
    plt.plot(train_x, y_hat)
    plt.show()



if __name__ == "__main__":
    #test_deep_first_order_polynomial()
    test_deep_sinusoidal_regression()