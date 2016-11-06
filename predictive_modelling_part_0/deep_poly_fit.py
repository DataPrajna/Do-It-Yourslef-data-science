from __future__ import print_function

from predictive_modelling_part_0.predictive_model import Model
from common.data_server import BatchDataServer
from common.model import TFBuildingBlocks


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


    def __init__(self, config):
        self.lr = config['learning_rate']
        self.n_samples = None
        self.num_epochs = config['num_epochs']
        self.order_poly = config['order_poly']
        self.print_frequency = config['print_frequency']

        self.x1d = tf.placeholder(shape=(None, 1), dtype=tf.float32)
        self.X = self.project_to_higher_dims(self.x1d)
        self.Y = tf.placeholder(shape=(None, 1), dtype=tf.float32)
        self.tensors = TFBuildingBlocks.create_sequence_of_xw_layers(self.X, config)


    def project_to_higher_dims(self, x):
        return tf.tile(x, tf.constant([1, self.order_poly], dtype=tf.int32)) ** tf.cast(
            tf.range(self.order_poly), dtype=tf.float32)

    def predictor(self):
        return self.tensors['y_hat']

    def predict(self, X=None):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            return sess.run([self.predictor()],
                            feed_dict={self.x1d: X})

    def error(self, X=None, Y=None):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            return sess.run([self.cost_function()],
                            feed_dict={self.x1d: X, self.Y: Y})

    def cost_function(self):
        return tf.reduce_sum(tf.pow(self.predictor() - self.Y, 2)) / (2 * self.n_samples)

    def solver(self):
        return tf.train.GradientDescentOptimizer(self.lr).minimize(self.cost_function())

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







if __name__ == '__main__':
    import doctest
    doctest.testmod()
