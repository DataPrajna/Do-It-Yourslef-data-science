from __future__ import print_function

from common.data_server import BatchDataServer
from common.model import TFBuildingBlocks
from common.h5_reader_writer import H5Writer


import tensorflow as tf
import numpy

class DeepPolyFit:
    """
     LinearRegressor class is designed to claculate weight (w) and bias (b) from a set of inputs and matching outputs.

     Args:
         lr  is learning rate
         num_epochs
         print_frequency

     Examples:

          let's create a traing and testing datasets

    """
    def __init__(self, config, X, Y):
        self.config = config
        self.lr = config['learning_rate']
        self.n_samples = None
        self.num_epochs = config['num_epochs']
        self.order_poly = config['order_poly']
        self.print_frequency = config['print_frequency']

        self.x1d = tf.placeholder(shape=(None, 1), dtype=tf.float32)
        self.X = self.project_to_higher_dims(self.x1d)
        self.Y = tf.placeholder(shape=(None, 1), dtype=tf.float32)
        self.tensors = TFBuildingBlocks.create_sequence_of_xw_layers(self.X, config)
        self.params = self.train(train_X= X, train_Y = Y)

    def project_to_higher_dims(self, x):
        return tf.tile(x, tf.constant([1, self.order_poly], dtype=tf.int32)) ** tf.cast(
            tf.range(self.order_poly), dtype=tf.float32)

    def predictor(self):
        return self.tensors['y_hat']

    def predict(self, X=None):
        self.update_tensors_with_learned_params(self.params)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            return sess.run([self.predictor()],
                            feed_dict={self.x1d: X})

    def update_tensors_with_learned_params(self, learned_params):
        self.tensors = TFBuildingBlocks.create_sequence_of_xw_layers_from_learned_params(self.X, learned_params, config=self.config)

    def error(self, learned_params, X=None, Y=None):
        self.update_tensors_with_learned_params(learned_params)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            return sess.run([self.cost_function()],
                            feed_dict={self.x1d: X, self.Y: Y})

    def cost_function(self):
        return tf.reduce_sum(tf.pow(self.predictor() - self.Y, 2)) / (2 * self.n_samples)

    def solver(self):
        return tf.train.GradientDescentOptimizer(self.lr).minimize(self.cost_function())

    def get_trained_params(self, sess):
        params_dict = dict()
        for key in self.tensors:
            if isinstance(self.tensors[key], tf.Variable):
                params_dict[key] = sess.run(self.tensors[key])

        return params_dict

    def write_params_to_file(self, filename, params_dict):
        H5Writer.write(filename, self.config, params_dict)

    def train(self, train_X=None, train_Y=None):
        self.n_samples = train_X.shape[0]
        cost_function = self.cost_function()
        solver = self.solver();
        batch_data = BatchDataServer(train_X, train_Y, batch_size = 10000)


        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            while batch_data.epoch < self.num_epochs:
                x1, y1 = batch_data.next()
                [_, cost] = sess.run([solver, cost_function],
                                         feed_dict={self.x1d: x1, self.Y: y1})

                if (batch_data.epoch + 1) % self.print_frequency == 0:
                    [cost] = sess.run([cost_function], feed_dict={self.x1d: train_X, self.Y: train_Y})
                    print('At Epoch {} the loss is {}'.format(batch_data.epoch, cost))
            params_dict = self.get_trained_params(sess)
        return params_dict












if __name__ == '__main__':
    import doctest
    doctest.testmod()
