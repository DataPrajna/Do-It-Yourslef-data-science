import tensorflow as tf
from collections import OrderedDict
import h5py
import json
import sys
sys.path.append("/home/amishra/appspace/Do-It-Yourslef-data-science/")
from common.data_server import BatchDataServer

from common.h5_reader_writer import H5Reader, H5Writer

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
    def fc_layer_ri(in_tensor, shape=[-1, 2], name='fc'):
        shape = [in_tensor.get_shape().as_list()[-1], shape[-1]]
        w = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.01))
        b = tf.Variable(tf.constant(0.1, shape=[shape[-1]]))
        conv = tf.matmul(in_tensor, w)
        bias = tf.nn.bias_add(conv, b)
        return bias, w, b

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
    def __init__(self, model_filename, x_ops, y_ops, bs=1, ih=224, iw=224, nc=3,  oh=2, ow=2):
        self.X = x_ops
        self.Y = y_ops
        self.print_frequency = 2
        self.num_epochs = 100
        self.learning_rate = 0.0001
        self.lr = 0.0001

        input_op = self.X
        self.node = OrderedDict({'iop': input_op})
        self.node['sub'] = self.node['iop'] - [103.939, 116.779, 123.68]
        config, self.wbp = H5Reader.read(model_filename)



    def ops_layers(self):
        self.node['c_1_1'] = TF_OPS.conv_layer(self.node['sub'], self.wbp['c_1_1.w'], self.wbp['c_1_1.b'])
        self.node['c_1_2'] = TF_OPS.conv_layer(self.node['c_1_1'], self.wbp['c_1_2.w'], self.wbp['c_1_2.b'])
        self.node['mp_1'] = TF_OPS.max_pool(self.node['c_1_2'])

        self.node['c_2_1'] = TF_OPS.conv_layer(self.node['mp_1'], self.wbp['c_2_1.w'], self.wbp['c_2_1.b'])
        self.node['c_2_2'] = TF_OPS.conv_layer(self.node['c_2_1'], self.wbp['c_2_2.w'], self.wbp['c_2_2.b'])
        self.node['mp_2'] = TF_OPS.max_pool(self.node['c_2_2'])

        self.node['c_3_1'] = TF_OPS.conv_layer(self.node['mp_2'], self.wbp['c_3_1.w'], self.wbp['c_3_1.b'])
        self.node['c_3_2'] = TF_OPS.conv_layer(self.node['c_3_1'], self.wbp['c_3_2.w'], self.wbp['c_3_2.b'])
        self.node['c_3_3'] = TF_OPS.conv_layer(self.node['c_3_2'], self.wbp['c_3_3.w'], self.wbp['c_3_3.b'])
        self.node['mp_3'] = TF_OPS.max_pool(self.node['c_3_3'])

        self.node['c_4_1'] = TF_OPS.conv_layer(self.node['mp_3'], self.wbp['c_4_1.w'], self.wbp['c_4_1.b'])
        self.node['c_4_2'] = TF_OPS.conv_layer(self.node['c_4_1'], self.wbp['c_4_2.w'], self.wbp['c_4_2.b'])
        self.node['c_4_3'] = TF_OPS.conv_layer(self.node['c_4_2'], self.wbp['c_4_3.w'], self.wbp['c_4_3.b'])
        self.node['mp_4'] = TF_OPS.max_pool(self.node['c_4_3'])

        self.node['c_5_1'] = TF_OPS.conv_layer(self.node['mp_4'], self.wbp['c_5_1.w'], self.wbp['c_5_1.b'])
        self.node['c_5_2'] = TF_OPS.conv_layer(self.node['c_5_1'], self.wbp['c_5_2.w'], self.wbp['c_5_2.b'])
        self.node['c_5_3'] = TF_OPS.conv_layer(self.node['c_5_2'], self.wbp['c_5_3.w'], self.wbp['c_5_3.b'])
        self.node['mp_5'] =  TF_OPS.faltten_layer(TF_OPS.max_pool(self.node['c_5_3']))

        self.node['fc_6'] = TF_OPS.fc_layer(self.node['mp_5'], self.wbp['fc_6.w'], self.wbp['fc_6.b'])
        self.node['fc_7'] = TF_OPS.fc_layer(self.node['fc_6'], self.wbp['fc_7.w'], self.wbp['fc_7.b'])
        self.node['fc_8'] = TF_OPS.fc_layer(self.node['fc_7'], self.wbp['fc_8.w'], self.wbp['fc_8.b'], is_relu=False)
        #self.node['fc_9'] = TF_OPS.fc_layer(self.node['fc_8'], self.wbp['fc_9.w'], self.wbp['fc_9.b'], is_relu=False)

        self.node['fc_9'], self.wbp['fc_9.w'], self.wbp['fc_9.b'] = TF_OPS.fc_layer_ri(self.node['fc_8'], shape =[-1, 1])

        #self.node['predictor'] = tf.nn.softmax(self.node['fc_8'], name="prob")

    def predict_an_image(self, path):
        import utils
        import numpy as np
        self.ops_layers()
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            rgb_scaled = utils.load_image(path)

            pred = sess.run([self.node['predictor']], feed_dict={self.node['iop']:rgb_scaled})
        utils.print_prob(pred[0][0])

    def cost_function_l2(self):
        return tf.reduce_mean(tf.pow(self.node['fc_9'] - self.Y, 2))

    def cost_function_l1(self):
        return tf.reduce_mean(tf.abs(tf.sub(self.node['fc_9'], self.Y)))

    def MAE(self):
        diff = self.cost_function_l1()
        true_val = tf.reduce_mean(self.Y)
        return tf.div(diff, true_val) * 100.0


    def solver(self):
        return tf.train.AdagradOptimizer(0.01).minimize(self.cost_function_l1())

    def evaluate(self, sess, bp):
        MAE_T = self.MAE()
        total_cost = 0
        import numpy as np
        from scipy import stats
        y_hat_T = self.node['fc_9']
        gt = []
        pred = []
        for b in range(len(bp)):
            x, y = bp.next()
            cost, y_hat = sess.run([MAE_T, y_hat_T], feed_dict={self.X: x, self.Y: y})
            if y.shape[0] == 7:
                y1 = np.reshape(y, [1, -1])
                y1_hat = np.reshape(y_hat, [1, -1])
                gt.append(y1)
                pred.append(y1_hat)
            total_cost += cost

        gt = np.asarray(gt)
        pred = np.asarray(pred)
        gt = np.reshape(gt, [1, -1])
        pred = np.reshape(pred, [1, -1])
        coe = np.corrcoef(gt, pred)
        print('person:', coe)
        srocc = stats.spearmanr(gt.tolist()[0], pred.tolist()[0])
        print('SROCC:', srocc)
        MAE = float(total_cost) / len(bp)
        return MAE

    def evaluate_standalone(self, bp):
        import numpy as np
        from scipy import stats
        MAE_T = self.MAE()
        y_hat_T = self.node['fc_9']
        total_cost = 0
        self.ops_layers()
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for b in range(len(bp)):
                x, y = bp.next()
                cost, y_hat = sess.run([MAE_T, y_hat_T], feed_dict={self.X: x, self.Y: y})
                y1 = np.reshape(y, [1, -1])
                y1_hat = np.reshape(y_hat, [1, -1])
                coe = np.corrcoef(y1, y1_hat)
                print('person:', coe)
                srocc = stats.spearmanr(y1, y1_hat)
                print('SROCC:', srocc)

                total_cost += cost[0]
            MAE = float(total_cost) / len(bp)
        return MAE

    @staticmethod
    def split_into_train_test_data_set(data_x, data_y, sf=0.8):
        train_num_sample = int(data_x.shape[0] * sf)
        X, Y = BatchDataServer.reset(data_x, data_y)
        return (X[:train_num_sample, :], Y[:train_num_sample, :]), (X[train_num_sample:, :], Y[train_num_sample:, :])

    @staticmethod
    def get_trained_params(sess, nodes):
        params_dict = dict()
        for key in nodes:
            if isinstance(nodes[key], tf.Variable):
                params_dict[key] = sess.run(nodes[key])
        return params_dict

    def write(self, sess, epoch, MAE, wbp):
        config = dict({
            'learning_rate': 0.0001,
            'num_epoch': 100,
            'MAE': 100})

        config['num_epoch'] = epoch
        config['MAE'] = MAE
        params = self.get_trained_params(sess, wbp)
        H5Writer.write('trained_diq_with_min_testing_error_step4.dah5', config=config, params_dict=params)
        print("Min error================================================{}".format(MAE))



    def train(self, X, Y, num_epoch, bs = 7):
        dtn, dtt = self.split_into_train_test_data_set(X, Y)
        self.ops_layers()
        tn_bp = BatchDataServer(dtn[0], dtn[1], batch_size=bs)
        tt_bp = BatchDataServer(dtt[0], dtt[1], batch_size=bs)

        solver = self.solver()
        min_test_mae = 120
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            test_mae = self.evaluate(sess, tt_bp)
            print('at epoch {}, initial_mae {}'.format(0, test_mae))
            for e in range(num_epoch):
                tn_bp.reset(tn_bp.X, tn_bp.Y)
                for b in range(len(tn_bp)):
                    x, y = tn_bp.next()
                    sess.run([solver], feed_dict={self.X:x, self.Y:y})
                    if b % 50 == 0:
                        train_mae = self.evaluate(sess, tn_bp)
                        print('at batch {}, train_mae {}'.format(b, train_mae))

                test_mae = self.evaluate(sess, tt_bp)
                if test_mae < min_test_mae:
                    min_test_mae = test_mae
                    self.write(sess, e, min_test_mae, self.wbp)
                print('at epoch {}, test_mae {}'.format(e, test_mae))





def run_train():
    from diq.misc import get_xy
    x, y = get_xy()
    print(x.shape, y.shape)
    x_op = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y_op = tf.placeholder(tf.float32, [None, 1])
    v = VGG('vgg15.dah5', x_op, y_op)
    v.train(x, y, 200)


def test():
    from diq.misc import get_xy
    x, y = get_xy()

    print(x.shape, y.shape)
    v = VGG('vgg15.dah5', bs=2)
    v.ops_layers()
    sv = v.solver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        [_, cost] = sess.run([sv, v.node['fc_9']],
                             feed_dict={v.X: x[0:2, :, :, :], v.Y: y[0:2, :]})
        print(cost)

if __name__=="__main__":
    run_train()
    #v.predict_an_image('cat.jpg')
