import numpy as np
from common.h5_reader_writer import H5Reader, H5Writer

def test_h5writer():
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
    params_dict = dict()
    params_dict['w1'] = np.asarray([1, 2, 3], dtype=np.float32)
    params_dict['w2'] = np.asarray([1, 6, 3], dtype=np.float32)
    params_dict['w3'] = np.asarray([4, 2, 3], dtype=np.float32)
    H5Writer.write('/tmp/test.h5', config, params_dict)

def test_h5reader():
    config, param_dict = H5Reader.read('/tmp/test.h5')
    print(config)
    print(param_dict)


if __name__=='__main__':
    test_h5writer()
    test_h5reader()



