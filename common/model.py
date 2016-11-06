import tensorflow as tf


class TFBuildingBlocks:
    def __init__(self):
        pass

    @staticmethod
    def multiply_xw(num_hidden_nodes, X, W=None):
        if W is None:
            X.get_shape().as_list()
            shape_bottom = X.get_shape().as_list()
            shape_current = [shape_bottom[-1], num_hidden_nodes]
            W = tf.Variable(tf.truncated_normal(shape=shape_current, stddev=0.1), dtype=tf.float32, name="w")
        current_tensor = tf.matmul(X, W)
        return current_tensor, W

    @staticmethod
    def create_sequence_of_xw_layers(input_x, config=None):
        tensors = dict()
        bottom = input_x
        tensors['input'] = bottom
        for layer in config['hidden_layers']:
            bottom, W = TFBuildingBlocks.multiply_xw(layer['shape'][-1], bottom)
            tensors[layer['op_name']] = bottom
            tensors[layer['var_name']] = W
        return tensors




