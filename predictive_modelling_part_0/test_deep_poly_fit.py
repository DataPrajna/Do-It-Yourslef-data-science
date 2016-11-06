
from predictive_modelling_part_0.deep_poly_fit import DeepPolyFit
import numpy




def test_first_order_polynomial():
    config = {
        'num_hidden_layers': 3,
        'order_poly': 4,
        'learning_rate': 0.01,
        'num_epochs': 5000,
        'print_frequency': 10,

        'hidden_layers': [
            {'op_name': 'h1', 'var_name': 'w1', 'shape': [-1, 35]},
            {'op_name': 'h2', 'var_name': 'w2', 'shape': [-1, 15]},
            {'op_name': 'h3', 'var_name': 'w3', 'shape': [-1, 7]},
            {'op_name': 'h4', 'var_name': 'w4', 'shape': [-1, 17]},
            {'op_name': 'y_hat', 'var_name': 'w5', 'shape': [-1, 1]}
        ]

    }
    lr = DeepPolyFit(config)
    train_x = numpy.linspace(-1,1,100,dtype = numpy.float32)
    print(train_x)
    train_x = train_x.reshape(-1,1)
    train_y = 5 + 2*train_x
    lr.train(train_X = train_x, train_Y = train_y)

def test_sinusoidal_regression():
    config = {
        'num_hidden_layers': 3,
        'order_poly': 5,
        'learning_rate': 0.01,
        'num_epochs': 50000,
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
    train_x = numpy.linspace(-3,3, 10000, dtype = numpy.float32)
    train_x = train_x.reshape(-1,1)
    train_y = numpy.sin(train_x) +  numpy.sin(3*train_x)
    lr.train(train_X = train_x, train_Y = train_y)

if __name__ == "__main__":
    #test_first_order_polynomial()
    test_sinusoidal_regression()