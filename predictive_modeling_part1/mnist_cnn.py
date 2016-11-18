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
    def __init__(self, config, learned_params=None):
        self.config = config
        self.lr = config['learning_rate']
        self.n_samples = None
        self.num_epochs = config['num_epochs']
        self.print_frequency = config['print_frequency']

        self.X = tf.placeholder(shape=(None, 784), dtype=tf.float32)
        self.p = tf.reshape(self.X, [-1, 28, 28, 1])
        self.Y = tf.placeholder(shape=(None, 10), dtype=tf.float32)
        self.tensors = dict()
        self.model(learned_params=learned_params)


    def get_learned_params(self, learned_params):
        for key in learned_params:
            self.tensors[key] = tf.Variable(learned_params[key])

    def initial_parameters(self):
        self.tensors['w1'] = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 32), stddev=0.1))
        self.tensors['b1'] = tf.Variable(tf.constant(0.1, shape=(1, 32)))

        self.tensors['w2'] = tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 64), stddev=0.1))
        self.tensors['b2'] = tf.Variable(tf.constant(0.1, shape=(1, 64)))

        self.tensors['w3'] = tf.Variable(tf.truncated_normal(shape=(7 * 7 * 64, 1024), stddev=0.1))
        self.tensors['b3'] = tf.Variable(tf.constant(0.1, shape=(1, 1024)))

        self.tensors['w4'] = tf.Variable(tf.truncated_normal(shape=(1024, 10), stddev=0.1))
        self.tensors['b4'] = tf.Variable(tf.constant(0.1, shape=(1, 10)))





    def model(self, learned_params=None):
        if learned_params == None:
            self.initial_parameters()
        else:
            self.get_learned_params(learned_params)

        self.tensors['conv1'] = tf.nn.conv2d(self.p, self.tensors['w1'], strides=[1, 1, 1, 1], padding='SAME')
        self.tensors['conv_plus_add1'] = tf.nn.relu(self.tensors['conv1'] + self.tensors['b1'])
        self.tensors['h1'] = tf.nn.max_pool(self.tensors['conv_plus_add1'], ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1], padding='SAME')

        self.tensors['conv2'] = tf.nn.conv2d(self.tensors['h1'], self.tensors['w2'], strides=[1, 1, 1, 1], padding='SAME')
        self.tensors['conv_plus_add2'] = tf.nn.relu(self.tensors['conv2'] + self.tensors['b2'])
        self.tensors['h2'] = tf.nn.max_pool(self.tensors['conv_plus_add2'], ksize=[1, 2, 2, 1],
                                            strides=[1, 2, 2, 1], padding='SAME')
        self.tensors['h2'] = tf.reshape(self.tensors['h2'], [-1, 7 * 7 * 64])

        self.tensors['h3'] = tf.nn.relu(tf.matmul(self.tensors['h2'], self.tensors['w3']) + self.tensors['b3'])
        if learned_params == None:
            self.keep_prob = tf.placeholder(tf.float32)
            self.tensors['drop_h3'] = tf.nn.dropout(self.tensors['h3'], self.keep_prob)
            self.tensors['y_hat'] = tf.nn.softmax(tf.matmul(self.tensors['drop_h3'], self.tensors['w4']) + self.tensors['b4'])
        else:
            self.tensors['y_hat'] = tf.nn.softmax(
                tf.matmul(self.tensors['h3'], self.tensors['w4']) + self.tensors['b4'])

        correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.predictor(), 1))
        self.tensors['accuracy'] = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predictor(self):
        return self.tensors['y_hat']

    def predict(self, X=None):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            return sess.run([self.predictor()],
                            feed_dict={self.X: X})

    def cost_function(self):
        return tf.reduce_mean(-tf.reduce_sum( self.Y * tf.log(self.predictor()), reduction_indices=[1]))

    def solver(self):
        return tf.train.AdamOptimizer(self.lr).minimize(self.cost_function())

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
        batch_data = BatchDataServer(train_X, train_Y, batch_size = 550)
        lr = self.lr

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            [accuracy] = sess.run([self.tensors['accuracy']],
                                  feed_dict={self.X: mnist.test.images, self.Y: mnist.test.labels, self.keep_prob:0.5})
            i=0;
            print("accuracy on testing images before training are", accuracy)
            while batch_data.epoch < self.num_epochs:
                x1, y1 = batch_data.next()
                [_, cost] = sess.run([solver, cost_function],
                                         feed_dict={self.X: x1, self.Y: y1,  self.keep_prob:0.5})
                if (batch_data.epoch + 1) % self.print_frequency == 0:
                    [cost] = sess.run([cost_function], feed_dict={self.X: x1, self.Y: y1, self.keep_prob:0.5})
                    print('At Epoch {} the loss is {}'.format(batch_data.epoch, cost))
                    batch_data.epoch = batch_data.epoch + 1
                    print(lr)


            params_dict = self.get_trained_params(sess)
            [accuracy] = sess.run([self.tensors['accuracy']],
                                  feed_dict={self.X: mnist.test.images, self.Y: mnist.test.labels, self.keep_prob: 1.0})
            self.config['learning_rate'] = lr
            print("accuracy on testing images after training is {} with a learning rate of {}".format(accuracy, lr))


        return params_dict






def test_train_mnist():
    config = {
        'num_hidden_layers': 4,
        'order_poly': 4,
        'learning_rate': 0.5,
        'num_epochs': 10,
        'print_frequency': 100,
    }

    train_x = mnist.train.images
    train_y = mnist.train.labels

    lr = MnistModel(config)
    learned_params =  lr.train(train_X = train_x, train_Y = train_y, filename='/tmp/sin.h5')
    y_hat = lr.predict(learned_params, X=mnist.test.images[0:1, :])
    print(numpy.argmax(y_hat[0], axis=1))






if __name__ == '__main__':
    test_train_mnist()
    #import doctest
    #doctest.testmod()

