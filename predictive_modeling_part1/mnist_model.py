import sys
sys.path.append('/home/amishra/appspace/Do-It-Yourslef-data-science/')

from common.data_server import BatchDataServer
from common.h5_reader_writer import H5Writer


import tensorflow as tf
import numpy
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
class MnistModel:
    """
     MnistModel class is designed to claculate weight (w) and bias (b) from a set of inputs and matching outputs.

     Args:
         lr  is learning rate
         num_epochs
         print_frequency

     Examples:

          let's create a traing and testing datasets

    """
    def __init__(self, config):
        self.config = config
        self.lr = config['learning_rate']
        self.n_samples = None
        self.num_epochs = config['num_epochs']
        self.print_frequency = config['print_frequency']

        self.X = tf.placeholder(shape=(None, 784), dtype=tf.float32)
        self.Y = tf.placeholder(shape=(None, 10), dtype=tf.float32)
        self.tensors = dict()
        self.model()


    def model(self):
        self.tensors['w1'] = tf.Variable(tf.truncated_normal(shape=(784, 10)), dtype=tf.float32)
        self.tensors['b1'] = tf.Variable(tf.truncated_normal(shape =(1,10)), dtype=tf.float32)
        self.tensors['y_hat'] = tf.nn.softmax(tf.add(tf.matmul(self.X, self.tensors['w1']), self.tensors['b1']))
        correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.predictor(), 1))
        self.tensors['accuracy'] = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predictor(self):
        return self.tensors['y_hat']

    def predict(self, learned_params, X=None):
        self.update_tensors_with_learned_params(learned_params)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            return sess.run([self.predictor()],
                            feed_dict={self.X: X})

    def update_tensors_with_learned_params(self, learned_params):
        self.tensors['w1'] = tf.Variable(learned_params['w1'], dtype=tf.float32)
        self.tensors['b1'] = tf.Variable(learned_params['b1'], dtype=tf.float32)
        self.tensors['y_hat'] = tf.nn.softmax(tf.add(tf.matmul(self.X, self.tensors['w1']), self.tensors['b1']))






    def cost_function(self):
        return tf.reduce_mean(-tf.reduce_sum( self.Y * tf.log(self.predictor()), reduction_indices=[1]))

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

    def train(self, train_X=None, train_Y=None, filename=None):
        self.n_samples = train_X.shape[0]
        cost_function = self.cost_function()
        solver = self.solver();
        batch_data = BatchDataServer(train_X, train_Y, batch_size = 1000)


        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            [accuracy] = sess.run([self.tensors['accuracy']],
                                  feed_dict={self.X: mnist.test.images, self.Y: mnist.test.labels})
            print("accuracy on testing images before training are", accuracy)
            while batch_data.epoch < self.num_epochs:
                x1, y1 = batch_data.next()
                [_, cost] = sess.run([solver, cost_function],
                                         feed_dict={self.X: x1, self.Y: y1})

                if (batch_data.epoch + 1) % self.print_frequency == 0:
                    [cost] = sess.run([cost_function], feed_dict={self.X: train_X, self.Y: train_Y})
                    print('At Epoch {} the loss is {}'.format(batch_data.epoch, cost))
                    batch_data.epoch = batch_data.epoch + 1

            params_dict = self.get_trained_params(sess)
            [accuracy] = sess.run([self.tensors['accuracy']],
                                  feed_dict={self.X: mnist.test.images, self.Y: mnist.test.labels})
            print("accuracy on testing images after training are", accuracy)


        return params_dict







def test_train_mnist():
    config = {
        'num_hidden_layers': 4,
        'order_poly': 4,
        'learning_rate': 0.5,
        'num_epochs': 1000,
        'print_frequency': 100,
    }

    train_x = mnist.train.images
    train_y = mnist.train.labels

    lr = MnistModel(config)
    learned_params =  lr.train(train_X = train_x, train_Y = train_y, filename='/tmp/sin.h5')






if __name__ == '__main__':
    test_train_mnist()
    #import doctest
    #doctest.testmod()

