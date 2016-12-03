import tensorflow as tf
from collections import OrderedDict
import h5py
import json
import sys
sys.path.append("/home/amishra/appspace/Do-It-Yourslef-data-science/")

from common.h5_reader_writer import H5Reader

class TF_OPS:
    @staticmethod
    def max_pool(in_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',  name='max_pool'):
        return tf.nn.max_pool(in_tensor, ksize=ksize, strides=strides,  padding=padding, name=name)

    @staticmethod
    def conv_layer(in_tensor, w, b, strides=[1, 1, 1, 1], padding='SAME',  name='conv'):
        conv = tf.nn.conv2d(in_tensor, w, strides=strides, padding=padding)
        bias = tf.nn.bias_add(conv, b)
        relu = tf.nn.relu(bias)
        return relu

    @staticmethod
    def fc_layer(in_tensor, w, b, is_relu=True, name='fc'):
        conv = tf.matmul(in_tensor, w)
        bias = tf.nn.bias_add(conv, b)
        if is_relu:
            bias = tf.nn.relu(bias)
        return bias

    @staticmethod
    def faltten_layer(in_tensor):
        shape = in_tensor.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        return tf.reshape(in_tensor, [-1, dim])




def translate(old_model, new_model):
    config = {

        'learning_rate': 0.0001,
        'num_epochs': 100,
        'print_frequency': 10,
    }
    js = [2, 2, 2, 3, 3, 3]
    fold = h5py.File(old_model)
    fnew = h5py.File(new_model, 'w')
    meta_group = fnew.create_group('meta')
    meta_group.attrs['config'] = json.dumps(config)
    data_group = fnew.create_group('data')
    for i in range(1, 6):
        for j in range(1, js[i]+1):
            name_w = 'c_{}_{}.w'.format(i, j)
            name_b = 'c_{}_{}.b'.format(i, j)
            nameo_w = 'conv_{}_{}.w'.format(i, j)
            nameo_b = 'conv_{}_{}.b'.format(i, j)
            data_group.create_dataset(name=name_w, data=fold[nameo_w])
            data_group.create_dataset(name=name_b, data=fold[nameo_b])

    for i in range(6, 9):
        name_w = 'fc_{}.w'.format(i)
        name_b = 'fc_{}.b'.format(i)
        data_group.create_dataset(name=name_w, data=fold[name_w])
        data_group.create_dataset(name=name_b, data=fold[name_b])

    fold.close()
    fnew.close()



class VGG:
    '''
    bs: batch size, ih=

    '''
    def __init__(self, model_filename, bs=1, ih=224, iw=224, nc=3,  oh=2, ow=2):
        input_op = tf.placeholder(shape=(bs, ih, iw, nc), dtype=tf.float32)
        self.tensors = OrderedDict({'iop': input_op})
        config, self.wbp = H5Reader.read(model_filename)
        VGG_MEAN = [103.939, 116.779, 123.68]
        self.tensors -= VGG_MEAN

    def ops_layers(self):
        self.tensors['c_1_1'] = TF_OPS.conv_layer(self.tensors['iop'],  self.wbp['c_1_1.w'], self.wbp['c_1_1.b'])
        self.tensors['c_1_2'] = TF_OPS.conv_layer(self.tensors['c_1_1'], self.wbp['c_1_2.w'],self.wbp['c_1_2.b'])
        self.tensors['mp_1'] = TF_OPS.max_pool(self.tensors['c_1_2'])

        self.tensors['c_2_1'] = TF_OPS.conv_layer(self.tensors['mp_1'], self.wbp['c_2_1.w'], self.wbp['c_2_1.b'])
        self.tensors['c_2_2'] = TF_OPS.conv_layer(self.tensors['c_2_1'], self.wbp['c_2_2.w'], self.wbp['c_2_2.b'])
        self.tensors['mp_2'] = TF_OPS.max_pool(self.tensors['c_2_2'])

        self.tensors['c_3_1'] = TF_OPS.conv_layer(self.tensors['mp_2'], self.wbp['c_3_1.w'], self.wbp['c_3_1.b'])
        self.tensors['c_3_2'] = TF_OPS.conv_layer(self.tensors['c_3_1'], self.wbp['c_3_2.w'], self.wbp['c_3_2.b'])
        self.tensors['c_3_3'] = TF_OPS.conv_layer(self.tensors['c_3_2'], self.wbp['c_3_3.w'], self.wbp['c_3_3.b'])
        self.tensors['mp_3'] = TF_OPS.max_pool(self.tensors['c_3_3'])

        self.tensors['c_4_1'] = TF_OPS.conv_layer(self.tensors['mp_3'], self.wbp['c_4_1.w'], self.wbp['c_4_1.b'])
        self.tensors['c_4_2'] = TF_OPS.conv_layer(self.tensors['c_4_1'], self.wbp['c_4_2.w'], self.wbp['c_4_2.b'])
        self.tensors['c_4_3'] = TF_OPS.conv_layer(self.tensors['c_4_2'], self.wbp['c_4_3.w'], self.wbp['c_4_3.b'])
        self.tensors['mp_4'] = TF_OPS.max_pool(self.tensors['c_4_3'])

        self.tensors['c_5_1'] = TF_OPS.conv_layer(self.tensors['mp_4'], self.wbp['c_5_1.w'], self.wbp['c_5_1.b'])
        self.tensors['c_5_2'] = TF_OPS.conv_layer(self.tensors['c_5_1'], self.wbp['c_5_2.w'], self.wbp['c_5_2.b'])
        self.tensors['c_5_3'] = TF_OPS.conv_layer(self.tensors['c_5_2'], self.wbp['c_5_3.w'], self.wbp['c_5_3.b'])
        self.tensors['mp_5'] =  TF_OPS.faltten_layer(TF_OPS.max_pool(self.tensors['c_5_3']))

        self.tensors['fc_6'] = TF_OPS.fc_layer(self.tensors['mp_5'], self.wbp['fc_6.w'], self.wbp['fc_6.b'])
        self.tensors['fc_7'] = TF_OPS.fc_layer(self.tensors['fc_6'], self.wbp['fc_7.w'], self.wbp['fc_7.b'])
        self.tensors['fc_8'] = TF_OPS.fc_layer(self.tensors['fc_7'], self.wbp['fc_8.w'],self.wbp['fc_8.b'], is_relu=False)
        self.tensors['predictor'] = tf.nn.softmax(self.tensors['fc_8'], name="prob")

    def predict_an_image(self, path):
        import utils
        import numpy as np
        self.ops_layers()
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            rgb_scaled = utils.load_image(path)

            pred = sess.run([self.tensors['predictor']], feed_dict={self.tensors['iop']:rgb_scaled})
        print("pred", pred[0][0])
        utils.print_prob(pred[0][0])




if __name__=="__main__":
    v = VGG('vgg15.dah5')
    v.predict_an_image('cat.jpg')
