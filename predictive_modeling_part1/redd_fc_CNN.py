import sys
sys.path.append(r'C:\Users\ppdash\workspace\Do-It-Yourslef-data-science')

from common.data_server import BatchDataServer
from common.h5_reader_writer import H5Writer
from da_vd_nilm import build_vgg16 as vgg16
import tensorflow as tf
import numpy
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt





class DeepNILM:
    """
     DeepNILM class is designed to claculate weight (w) and bias (b) from a set of inputs and matching outputs.

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
        self.feature_dimension = 400
        self.number_class = 8

        self.Y = tf.placeholder(shape=(None, self.number_class), dtype=tf.float32)
        vgg_graph = vgg16(init_node=tf.placeholder(tf.float32, [None, 1,  self.feature_dimension, 1]))
        self.X  = vgg_graph['tn']['place_holder']
        self.tensors = vgg_graph['tn']
        self.variables = vgg_graph['var']
        self.model()

    def model(self):
        self.tensors['y_hat'] = tf.nn.softmax(self.tensors['fc_3'])
        correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.predictor(), 1))
        self.tensors['accuracy'] = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predictor(self):
        return self.tensors['y_hat']

    def predict(self, learned_params, X=None):
        self.model();
        with tf.Session() as sess:
            self.update_tensors_with_learned_params(learned_params, sess)
            sess.run(tf.global_variables_initializer())
            return sess.run([self.predictor()],
                            feed_dict={self.X: X})

    def compute_stats(self, x, y, sess):
        batch_data = BatchDataServer(x, y, batch_size=128)
        total_accuracy = 0
        while batch_data.epoch < len(batch_data):
            x1, y1 = batch_data.next()
            [accuracy] = sess.run([self.tensors['accuracy']],
                                       feed_dict={self.X: x1, self.Y: y1})
            total_accuracy = total_accuracy + accuracy
        return total_accuracy / len(batch_data)

    def update_tensors_with_learned_params(self, learned_params, sess):
        for key in learned_params:
            sess.run(self.variables[key].assign(learned_params[key]))

    def cost_function(self):
        return tf.reduce_mean(-tf.reduce_sum( self.Y * tf.log(self.predictor()+ 1e-10), reduction_indices=[1]))

    def solver(self):
        return tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam').minimize(self.cost_function())


    def get_trained_params(self, sess):
        return self.variables

    def confusion_mat(self, x, y):
        res = tf.stack([tf.argmax(self.Y, 1), tf.argmax(self.predictor(), 1)])
        ans = res.eval(feed_dict={self.X: x, self.Y: y})
        confusion = np.zeros([self.number_class, self.number_class], int)
        for p in ans.T:
            confusion[p[0], p[1]] += 1

        dataframe =  pd.DataFrame(confusion)
        label = ['dw', 'fr', 'li', 'wd', 'mw', 'ket',
                 'comp', 'oven']
        dataframe.index = label
        dataframe.columns = label

        print(pd.DataFrame(dataframe))
        return pd.DataFrame(dataframe)

    def write_params_to_file(self, filename, params_dict):
        H5Writer.write(filename, self.config, params_dict)

    def train(self, x=None, y=None, filename=None):
        cost_function = self.cost_function()
        solver = self.solver();
        batch_data = BatchDataServer(x, y, batch_size = 128)
        lr = self.lr
        loss = []
        accuracy_training =[]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.confusion_mat(x, y)
            accuracy = self.compute_stats(x, y, sess)
            print("accuracy on testing appliance data before training are", accuracy)
            while batch_data.epoch < self.num_epochs:
                x1, y1 = batch_data.next()
                [_, cost, accuracy] = sess.run([solver, cost_function, self.tensors['accuracy']],
                                         feed_dict={self.X: x1, self.Y: y1,  self.learning_rate:lr})
                loss.append(cost)
                accuracy_training.append(accuracy)

                if (batch_data.epoch + 1) % self.print_frequency == 0:
                    print('At Epoch {} the loss is {}'.format(batch_data.epoch, cost))

                    batch_data.epoch = batch_data.epoch + 1
                    lr = self.lr * numpy.exp(-3.1*(batch_data.epoch/(self.num_epochs+1.0)))
                    print(lr)
                    params_dict = self.get_trained_params(sess)
                    frame = self.confusion_mat(x, y)
                    accuracy = self.compute_stats(x, y, sess)
                    self.config['learning_rate'] = lr
                    print("accuracy on testing appliance data after training is {} with a learning rate of {}".format(accuracy, lr))
                    frame = frame / frame.sum()
                    if accuracy > 0.7:
                        print('writing to')
                        frame.to_csv('confusion.csv')

        return_dict = {'parameters': params_dict, 'loss': loss, 'Accuracy': accuracy_training}
        return return_dict





import numpy as np
import random

def test_train_NILM(n_sample = None):
    config = {
        'num_hidden_layers': 5,
        'order_poly': 4,
        'learning_rate': 0.00001,
        'num_epochs': 1000,
        'print_frequency': 100,
    }

    lr = DeepNILM(config)
    f = h5py.File(r'C:\Users\ppdash\workspace\deep-nilmtk\Data\b1to5top5ukDale.h5')
    x = f['samples'].value

    n_sample = int(x.shape[0] * 0.8)
    rIdx = random.sample(range(x.shape[0]), x.shape[0]);
    print(rIdx)
    x = x / np.max(x) - 0.5
    x = x[rIdx]
    x = x.reshape((-1, 1, 400, 1))
    train_x = x[0:n_sample,:]
    test_x = x[n_sample:,:]
    dummy = f['labels'].value
    dummy = dummy[rIdx]
    train_y = np.zeros(shape=(n_sample, lr.number_class))
    for i in range(0, n_sample):
        train_y[i, dummy[i]] = 1
    test_dummy = dummy[n_sample:]

    learned_params = lr.train(x=train_x, y=train_y, filename=None)
   #  y_hat = lr.predict(learned_params['parameters'], X = test_x)
   #  test_predict = np.argmax(y_hat[0], axis=1)
   #  confusion_test = np.zeros([lr.number_class,lr.number_class], int)
   #
   #  for i in range(len(test_predict)):
   #      confusion_test[test_dummy[i],test_predict[i]] += 1
   #  dataframe_test = pd.DataFrame(confusion_test)
   #  label = ['dw', 'fr', 'li', 'wd', 'mw', 'ket',
   #               'comp', 'oven']
   #  dataframe_test.index = label
   #  dataframe_test.columns = label
   #  print('So test result is:')
   #  print(pd.DataFrame(dataframe_test))
   #  dataframe_test["sum"] = dataframe_test.sum(axis = 1)
   #  dataframe_test_final = dataframe_test.loc[:,"dw":"oven"].div(dataframe_test["sum"], axis = 0)
   #  print(np.round(dataframe_test_final,3))
   #  true_match = 0;
   #  # for i in range(len(test_predict)):
   #  #     if test_dummy[i] == test_predict[i]:
   #  #         true_match += 1
   #
   #  correct_prediction_test = np.float32(np.equal(test_dummy.reshape(-1,1),test_predict.reshape(-1,1)))
   #  #print(correct_prediction_test)
   #  accuracy_test = np.mean(correct_prediction_test)
   # # accuracy_test = true_match / len(test_predict)
   #  print("accuracy on testing appliance data after training is {}".format(accuracy_test))
   #  loss = learned_params['loss']
   #  accuracy_training = learned_params['Accuracy']
   #  plt.plot(loss)
   #  plt.xlabel('epoch')
   #  plt.ylabel('loss')
   #  plt.title('Loss in Training Phase')
   #  plt.show()
   #  plt.plot(accuracy_training)
   #  plt.xlabel('epoch')
   #  plt.ylabel('Accuracy')
   #  plt.title('Accuracy in Training Phase')
   #  plt.show()
   #
   #
   #  #plt.plot(accuracy_test)
   #  #plt.show()
   #

if __name__ == '__main__':
    test_train_NILM()
    #import doctest
    #doctest.testmod()




