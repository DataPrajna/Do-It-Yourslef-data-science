import tensorflow as tf


class TFBuildingBlocks:
    def __init__(self):
        pass

    @staticmethod
    def multiply_xw(num_hidden_nodes, X, W=None):
        if W is None:
            shape_bottom = X.get_shape().as_list()
            W = tf.truncated_normal(shape=[shape_bottom[-1], num_hidden_nodes], stddev=0.1)
        W_tensor = tf.Variable(W, dtype=tf.float32, name="w")
        current_tensor = tf.matmul(X, W_tensor)
        return current_tensor, W_tensor

    @staticmethod
    def multiply_xw_add_b(num_hidden_nodes, X, W=None, b=None):
        if W is None:
            shape_bottom = X.get_shape().as_list()
            W = tf.truncated_normal(shape=[shape_bottom[-1], num_hidden_nodes], stddev=0.1)
        if b is None:
            shape_bottom = X.get_shape().as_list()
            W = tf.truncated_normal(shape=[shape_bottom[-1], num_hidden_nodes], stddev=0.1)
            b = tf.truncated_normal(shape=[1, num_hidden_nodes], stddev=0.1)
        W_tensor = tf.Variable(W, dtype=tf.float32, name="w")
        b_tensor = tf.Variable(b, dtype=tf.float32, name="w")

        current_tensor = tf.add(tf.matmul(X, W_tensor), b)
        return current_tensor, W_tensor

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

    @staticmethod
    def create_sequence_of_xw_b_layers(input_x, config=None):
        tensors = dict()
        bottom = input_x
        tensors['input'] = bottom
        for layer in config['hidden_layers']:
            bottom, W = TFBuildingBlocks.multiply_xw_add_b(layer['shape'][-1], bottom)
            tensors[layer['op_name']] = bottom
            tensors[layer['var_name']] = W
        return tensors

    @staticmethod
    def create_sequence_of_xw_layers_from_learned_params(input_x, learned_params, config=None):
        tensors = dict()
        bottom = input_x
        tensors['input'] = bottom
        for layer in config['hidden_layers']:
            bottom, W = TFBuildingBlocks.multiply_xw(layer['shape'][-1], bottom, W=learned_params[layer['var_name']])
            tensors[layer['op_name']] = bottom
            tensors[layer['var_name']] = W
        return tensors




