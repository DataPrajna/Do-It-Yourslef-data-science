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
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.n_samples = None
        self.num_epochs = config['num_epochs']
        self.print_frequency = config['print_frequency']

        self.X = tf.placeholder(shape=(None, 784), dtype=tf.float32)
        self.Y = tf.placeholder(shape=(None, 10), dtype=tf.float32)
        self.node = dict()
        self.model()


    def model(self):
        self.node['w1'] = tf.Variable(tf.truncated_normal(shape=(784, 30), stddev=0.001))
        self.node['b1'] = tf.Variable(tf.truncated_normal(shape =(1, 30), stddev=0.001))
        self.node['h1'] = tf.nn.relu(tf.add(tf.matmul(self.X, self.node['w1']), self.node['b1']))

        self.node['w2'] = tf.Variable(tf.truncated_normal(shape=(30, 20), stddev=0.001))
        self.node['b2'] = tf.Variable(tf.truncated_normal(shape=(1, 20), stddev=0.001))
        self.node['h2'] = tf.nn.relu(tf.add(tf.matmul(self.node['h1'], self.node['w2']), self.node['b2']))


        self.node['w3'] = tf.Variable(tf.truncated_normal(shape=(20, 10), stddev=0.01))
        self.node['b3']= tf.Variable(tf.truncated_normal(shape=(1, 10), stddev=0.01))


        self.node['y_hat'] = tf.nn.softmax(tf.add(tf.matmul(self.node['h2'], self.node['w3']), self.node['b3']))
        correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.predictor(), 1))
        self.node['accuracy'] = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predictor(self):
        return self.node['y_hat']

    def predict(self, learned_params, X=None):
        self.update_tensors_with_learned_params(learned_params)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            return sess.run([self.predictor()],
                            feed_dict={self.X: X})

    def update_tensors_with_learned_params(self, learned_params):

        self.node['w1'] = tf.Variable(learned_params['w1'])
        self.node['b1'] = tf.Variable(learned_params['b1'])
        self.node['h1'] = tf.nn.relu(tf.add(tf.matmul(self.X, self.node['w1']), self.node['b1']))

        self.node['w2'] = tf.Variable(learned_params['w2'])
        self.node['b2'] = tf.Variable(learned_params['b2'])
        self.node['h2'] = tf.nn.relu(tf.add(tf.matmul(self.node['h1'], self.node['w2']), self.node['b2']))
        self.node['w3'] = tf.Variable(learned_params['w3'])
        self.node['b3'] = tf.Variable(learned_params['b3'])

        self.node['y_hat'] = tf.nn.softmax(
            tf.add(tf.matmul(self.node['h2'], self.node['w3']), self.node['b3']))

    def cost_function(self):
        return tf.reduce_mean(-tf.reduce_sum( self.Y * tf.log(self.predictor()), reduction_indices=[1]))

    def solver(self):
        return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost_function())

    def get_trained_params(self, sess):
        params_dict = dict()
        for key in self.node:
            if isinstance(self.node[key], tf.Variable):
                params_dict[key] = sess.run(self.node[key])

        return params_dict

    def write_params_to_file(self, filename, params_dict):
        H5Writer.write(filename, self.config, params_dict)

    def train(self, train_X=None, train_Y=None, filename=None):
        self.n_samples = train_X.shape[0]
        cost_function = self.cost_function()
        solver = self.solver();
        batch_data = BatchDataServer(train_X, train_Y, batch_size = 1000)
        lr = self.lr

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            [accuracy] = sess.run([self.node['accuracy']],
                                  feed_dict={self.X: mnist.test.images, self.Y: mnist.test.labels})
            print("accuracy on testing images before training are", accuracy)
            while batch_data.epoch < self.num_epochs:
                x1, y1 = batch_data.next()
                [_, cost] = sess.run([solver, cost_function],
                                         feed_dict={self.X: x1, self.Y: y1,  self.learning_rate:lr})
                if (batch_data.epoch + 1) % self.print_frequency == 0:
                    [cost] = sess.run([cost_function], feed_dict={self.X: train_X, self.Y: train_Y, self.learning_rate:lr})
                    print('At Epoch {} the loss is {}'.format(batch_data.epoch, cost))
                    batch_data.epoch = batch_data.epoch + 1
                    lr = self.lr * numpy.exp(-3.0*(batch_data.epoch/(self.num_epochs+1.0)))
                    print(lr)


            params_dict = self.get_trained_params(sess)
            [accuracy] = sess.run([self.node['accuracy']],
                                  feed_dict={self.X: mnist.test.images, self.Y: mnist.test.labels})
            self.config['learning_rate'] = lr
            print("accuracy on testing images after training is {} with a learning rate of {}".format(accuracy, lr))


        return params_dict







def test_train_mnist():
    config = {
        'num_hidden_layers': 4,
        'order_poly': 4,
        'learning_rate': 0.5,
        'num_epochs': 100,
        'print_frequency': 10,
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

