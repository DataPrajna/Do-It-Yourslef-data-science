
from __future__ import print_function

from predictive_modelling_part_0.predictive_model import Model
from common.data_server import BatchDataServer


import tensorflow as tf
import numpy

class LinearRegressor:
    """
     LinearRegressor class is designed to claculate weight (w) and bias (b) from a set of inputs and matching outputs.

     Args:
         lr  is learning rate
         num_epochs
         print_frequency

     Examples:
          let's create a traing and testing datasets
          >>> num_sample = 10000
          >>> train_X = numpy.linspace(-3, 3, num_sample, dtype=numpy.float32)
          >>> train_X = train_X.reshape((-1, 1))
          >>> train_Y = numpy.sin(train_X)
          >>> train_Y = train_Y.reshape(-1, 1)

          now lets instantiate the class
          >>> lr = LinearRegressor(lr = 0.001, num_epocs = 5000, print_frequency = 100, num_features=5)

          let set the initial W and b using a random numb
          >>> lr.set_parameters()


           now lets train the linear regressor on the constructor training dataset train_x and train_y

          >>> W = lr.train(train_X = train_X, train_Y = train_Y)

          now reset the w, b parameters with the learned w and b

          >>> lr.set_parameters(W = W)

          now predict y for a given x

          >>> y_hat = lr.predict(X=train_X)

          >>> print(W)

    """

    def __init__(self, lr=0.01, num_epocs=10000, print_frequency=100, num_features=1):
        self.lr = lr
        self.n_samples = None
        self.num_epocs = num_epocs
        self.num_features = num_features
        self.print_frequency = print_frequency
        self.x1d = tf.placeholder(shape=(None, 1), dtype=tf.float32)
        self.X = self.project_to_higher_dims(self.x1d)
        self.Y = tf.placeholder(shape=(None, 1), dtype=tf.float32)

    def project_to_higher_dims(self, x):
        return tf.tile(x, tf.constant([1, self.num_features], dtype=tf.int32)) ** tf.cast(
            tf.range(self.num_features), dtype=tf.float32)

    def set_parameters(self, W=None):
        if W is None:
            self.W = tf.Variable(tf.truncated_normal(shape=(self.num_features,1), stddev=0.1), dtype=tf.float32, name="w")
        else:
            self.W = tf.Variable(W, name="W")


    def predictor(self):
        return tf.matmul(self.X, self.W)

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
            while batch_data.epoch < self.num_epocs:
                x1, y1 = batch_data.next()
                [_, cost] = sess.run([solver, cost_function],
                                         feed_dict={self.x1d: x1, self.Y: y1})

                if (batch_data.epoch + 1) % self.print_frequency == 0:
                    [cost] = sess.run([cost_function], feed_dict={self.x1d: train_X, self.Y: train_Y})
                    print('At Epoch {} the loss is {}'.format(batch_data.epoch, cost))
                    print('w {} '.format(sess.run(self.W)))

            return sess.run(self.W)

if __name__ == '__main__':
    import doctest
    doctest.testmod()



























