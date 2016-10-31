#from predictive_modelling_part_0.predictive_model import Model

from __future__ import print_function

import tensorflow as tf
import numpy


class LinearRegressor:

    """
     LinearRegressor class is designed to claculate weight (w) and bias (b) from a set of inputs and matching outputs.

     Args:
         lr  is learning rate
         num_epocs
         print_frequency

     Examples:

          let's create a traing and testing datasets
          >>> train_X = numpy.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
          >>> train_Y = numpy.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

          now lets instantiate the class

          >>> lr = LinearRegressor(lr = 0.01, num_epocs = 1000, print_frequency = 1001)

          let set the initial W and b using a random number

           >>> lr.set_wb_parameter(W = numpy.random.randn(), b = numpy.random.randn())

           now lets train the linear regressor on the constructor training dataset train_x and train_y

          >>> W, b = lr.train(train_X = train_X, train_Y = train_Y)

          now reset the w, b parameters with the learned w and b

          >>> lr.set_wb_parameter(W = W, b = b)

          now predict y for a given x

          >>> y_hat = lr.predict(X=[3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])





    """
    def __init__(self, lr = 0.01, num_epocs = 10000, print_frequency = 100):
        self.lr = lr
        self.n_samples = None
        self.num_epocs = num_epocs
        self.print_frequency = print_frequency
        self.X = tf.placeholder('float')
        self.Y = tf.placeholder('float')

    def set_wb_parameter(self, W=None, b=None):

        if W is None:
            self.W = tf.Variable(numpy.random.randn(), name="w")
        else:
            self.W = tf.Variable(W, name = "W")
        if b is None:
            self.b = tf.Variable(numpy.random.randn(), name="b")
        else:
            self.b = tf.Variable(b, name = "b")

    def predictor(self):
        return tf.add(tf.mul(self.X, self.W), self.b)

    def predict(self, X=None, Y=None):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            return sess.run([self.predictor()],
                     feed_dict={self.X: X})

    def error(self, X=None, Y=None):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            return sess.run([self.cost_function()],
                            feed_dict={self.X: X, self.Y:Y})


    def cost_function(self):
         return tf.reduce_sum(tf.pow(self.predictor() - self.Y, 2)) / (2 * self.n_samples)

    def solver(self):
        return tf.train.GradientDescentOptimizer(self.lr).minimize(self.cost_function())

    def train(self, train_X = None, train_Y = None):
        self.n_samples = train_X.shape[0]
        cost_function = self.cost_function()
        solver =  self.solver();
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for epoch in range(self.num_epocs):
                for x, y in zip(train_X, train_Y):
                    [_, cost] = sess.run([solver, cost_function],
                                         feed_dict={self.X : x, self.Y : y})

                if (epoch +1) % self.print_frequency == 0:
                    [cost] = sess.run([cost_function], feed_dict={self.X : train_X, self.Y : train_Y})
                    print('At Epoch {} the loss is {}'.format(epoch, cost))
                    print('w {} and b {}'.format(sess.run(self.W), sess.run(self.b)))

            return sess.run(self.W), sess.run(self.b)


def test_linear_regressor():
    lr = LinearRegressor(lr=0.01, num_epocs=1000, print_frequency=100)
    train_X = numpy.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    train_Y = numpy.asarray( [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
    lr.set_wb_parameter(W=numpy.random.randn(), b=numpy.random.randn())
    W, b = lr.train(train_X=train_X, train_Y=train_Y)
    lr.set_wb_parameter(W=W, b=b)
    y_hat = lr.predict(X=[3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    error = lr.error(
        X=[3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1],
        Y = [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
    print(y_hat)
    print(error)


if __name__=='__main__':
    import doctest

    doctest.testmod()
    #test_linear_regressor()









































#
#
# # Launch the graph
# with tf.Session() as sess:
#     sess.run(init)
#
#     # Fit all training data
#     for epoch in range(training_epochs):
#         for (x, y) in zip(train_X, train_Y):
#             sess.run(optimizer, feed_dict={X: x, Y: y})
#
#         # Display logs per epoch step
#         if (epoch+1) % display_step == 0:
#             c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
#             print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
#                 "W=", sess.run(W), "b=", sess.run(b))
#
#     print("Optimization Finished!")
#     training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
#     print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
#
#     # Graphic display
#     plt.plot(train_X, train_Y, 'ro', label='Original data')
#     plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
#     plt.legend()
#     plt.show()
#
#     # Testing example, as requested (Issue #2)
#     test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
#     test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])
#
#     print("Testing... (Mean square loss Comparison)")
#     testing_cost = sess.run(
#         tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
#         feed_dict={X: test_X, Y: test_Y})  # same function as cost above
#     print("Testing cost=", testing_cost)
#     print("Absolute mean square loss difference:", abs(
#         training_cost - testing_cost))
#
#     plt.plot(test_X, test_Y, 'bo', label='Testing data')
#     plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
#     plt.legend()
#     plt.show()
#
#
