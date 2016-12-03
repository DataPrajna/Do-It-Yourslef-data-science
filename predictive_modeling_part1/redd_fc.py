import sys
sys.path.append('/home/amishra/appspace/Do-It-Yourslef-data-science/')

from common.data_server import BatchDataServer
from common.h5_reader_writer import H5Writer


import tensorflow as tf
import numpy
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import h5py
import numpy as np
import pandas as pd





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

        self.X = tf.placeholder(shape=(None, 400), dtype=tf.float32)
        self.Y = tf.placeholder(shape=(None, 6), dtype=tf.float32)
        self.tensors = dict()
        self.model()


    def model(self):
        self.tensors['w1'] = tf.Variable(tf.truncated_normal(shape=(400, 200), stddev=0.01))
        self.tensors['b1'] = tf.Variable(tf.truncated_normal(shape =(1,200), stddev=0.01))
        self.tensors['h1'] = tf.nn.relu(tf.add(tf.matmul(self.X, self.tensors['w1']), self.tensors['b1']))

        self.tensors['w2'] = tf.Variable(tf.truncated_normal(shape=(200, 200), stddev=0.01))
        self.tensors['b2'] = tf.Variable(tf.truncated_normal(shape=(1, 200), stddev=0.01))
        self.tensors['h2'] = tf.nn.relu(tf.add(tf.matmul( self.tensors['h1'], self.tensors['w2']), self.tensors['b2']))


        self.tensors['w3'] = tf.Variable(tf.truncated_normal(shape=(200, 100), stddev=0.01))
        self.tensors['b3'] = tf.Variable(tf.truncated_normal(shape=(1, 100), stddev=0.01))
        self.tensors['h3'] = tf.nn.relu(tf.add(tf.matmul( self.tensors['h2'], self.tensors['w3']), self.tensors['b3']))

        self.tensors['w4'] = tf.Variable(tf.truncated_normal(shape=(100, 6), stddev=0.01))
        self.tensors['b4']= tf.Variable(tf.truncated_normal(shape=(1, 6), stddev=0.01))


        self.tensors['y_hat'] = tf.nn.softmax(tf.add(tf.matmul(self.tensors['h3'], self.tensors['w4']), self.tensors['b4']))
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

        self.tensors['w1'] = tf.Variable(learned_params['w1'])
        self.tensors['b1'] = tf.Variable(learned_params['b1'])
        self.tensors['h1'] = tf.nn.relu(tf.add(tf.matmul(self.X, self.tensors['w1']), self.tensors['b1']))

        self.tensors['w2'] = tf.Variable(learned_params['w2'])
        self.tensors['b2'] = tf.Variable(learned_params['b2'])
        self.tensors['h2'] = tf.nn.relu(tf.add(tf.matmul(self.tensors['h1'], self.tensors['w2']), self.tensors['b2']))
        self.tensors['w3'] = tf.Variable(learned_params['w3'])
        self.tensors['b3'] = tf.Variable(learned_params['b3'])

        self.tensors['y_hat'] = tf.nn.softmax(
            tf.add(tf.matmul(self.tensors['h2'], self.tensors['w3']), self.tensors['b3']))

    def cost_function(self):
        return tf.reduce_mean(-tf.reduce_sum( self.Y * tf.log(self.predictor()+ 1e-10), reduction_indices=[1]))

    def solver(self):
        return tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam').minimize(self.cost_function())


    def get_trained_params(self, sess):
        params_dict = dict()
        for key in self.tensors:
            if isinstance(self.tensors[key], tf.Variable):
                params_dict[key] = sess.run(self.tensors[key])

        return params_dict

    def confusion_mat(self, x, y):
        res = tf.pack([tf.argmax(self.Y, 1), tf.argmax(self.predictor(), 1)])
        ans = res.eval(feed_dict={self.X: x, self.Y: y})
        confusion = np.zeros([6, 6], int)
        for p in ans.T:
            confusion[p[0], p[1]] += 1

        dataframe =  pd.DataFrame(confusion)
        label = ['dish washer', 'fridge', 'light', 'washer dryer', 'microwave', 'sockets']
        dataframe.index = label
        dataframe.columns = label

        print(pd.DataFrame(dataframe))
        return pd.DataFrame(dataframe)

    def write_params_to_file(self, filename, params_dict):
        H5Writer.write(filename, self.config, params_dict)

    def train(self, x=None, y=None, filename=None):
        self.n_samples = 2476
        train_X = x[0:self.n_samples, :]
        train_Y = y[0:self.n_samples, :]

        test_X = train_X
        test_Y = train_Y


        cost_function = self.cost_function()
        solver = self.solver();
        batch_data = BatchDataServer(train_X, train_Y, batch_size = 128)
        lr = self.lr

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            [accuracy] = sess.run([self.tensors['accuracy']],
                                  feed_dict={self.X: train_X, self.Y: train_Y})
            self.confusion_mat(train_X, train_Y)
            print("accuracy on testing images before training are", accuracy)
            while batch_data.epoch < self.num_epochs:
                x1, y1 = batch_data.next()
                [_, cost] = sess.run([solver, cost_function],
                                         feed_dict={self.X: x1, self.Y: y1,  self.learning_rate:lr})
                if (batch_data.epoch + 1) % self.print_frequency == 0:
                    [cost] = sess.run([cost_function], feed_dict={self.X: train_X, self.Y: train_Y, self.learning_rate:lr})
                    print('At Epoch {} the loss is {}'.format(batch_data.epoch, cost))
                    batch_data.epoch = batch_data.epoch + 1
                    lr = self.lr * numpy.exp(-3.1*(batch_data.epoch/(self.num_epochs+1.0)))
                    print(lr)
                    params_dict = self.get_trained_params(sess)
                    [accuracy] = sess.run([self.tensors['accuracy']],
                                          feed_dict={self.X: test_X, self.Y: test_Y})
                    self.config['learning_rate'] = lr
                    print("accuracy on testing images after training is {} with a learning rate of {}".format(accuracy, lr))
                    frame = self.confusion_mat(test_X, test_Y)
                    frame = frame / frame.sum()
                    if accuracy > 0.7:
                        print('writing to')
                        frame.to_csv('confusion.csv')


        return params_dict





import numpy as np

def test_train_mnist():
    config = {
        'num_hidden_layers': 4,
        'order_poly': 4,
        'learning_rate': 0.0001,
        'num_epochs': 10000,
        'print_frequency': 100,
    }
    f = h5py.File('/home/amishra/appspace/deep-nilmtk/common/b1to5top5.h5')
    train_x = f['samples'].value
    train_x = train_x / np.max(train_x) - 0.5
    dummy = f['labels'].value
    train_y = np.zeros(shape=(2476, 6))
    for i in range(0, 2476):
        train_y[i, dummy[i]] = 1

    lr = MnistModel(config)
    learned_params =  lr.train(x = train_x, y = train_y, filename='/tmp/sin.h5')
    #y_hat = lr.predict(learned_params, X=mnist.test.images[0:1, :])
    #print(numpy.argmax(y_hat[0], axis=1))






if __name__ == '__main__':
    test_train_mnist()
    #import doctest
    #doctest.testmod()




