

import threading
import sys
sys.path.append("/home/amishra/appspace/Do-It-Yourslef-data-science/")

import tensorflow as tf
from common.data_server import BatchDataServer
from diq.vgg16 import VGG
from common.h5_reader_writer import H5Writer


class SessionManager:
    def __init__(self,bs=1, ih=224, iw=224, m_qs=1, nc=3, n_class=1):
        self.x_ph = tf.placeholder(tf.float32, [None, ih, iw, nc])
        self.y_ph = tf.placeholder(tf.float32, [None, 1 * n_class])

        self.q = tf.FIFOQueue(m_qs, [tf.float32, tf.float32],
                              shapes=[[ih, iw, nc], [1 * n_class]])

        self.enqueue_op = self.q.enqueue_many([self.x_ph, self.y_ph])

        self.bs = tf.Variable(bs)
        self.x, self.y = self.q.dequeue_many(self.bs)

        self.sess = tf.Session()


class TFBatchDataProvider(BatchDataServer):
    def __init__(self, sm, data_x, data_y, bs=1):
        super().__init__(data_x, data_y, batch_size=bs)
        self.sm = sm
        self.num_samples = data_x.shape[0]
        self.stop_providing = False

    def provider(self):
        while not self.stop_providing:
            for b in range(len(self)):
                x, y = self.next()
                self.sm.sess.run(self.sm.enqueue_op, feed_dict={self.sm.x_ph: x, self.sm.y_ph: y})

    def last_batch_size(self):
        last_batch = self.num_samples % self.batch_size
        return last_batch if last_batch != 0 else self.batch_size

def split_into_train_test_data_set(data_x, data_y, sf=0.6):
    train_num_sample = int(data_x.shape[0] * sf)
    X, Y = BatchDataServer.reset(data_x, data_y)
    return (X[:train_num_sample, :], Y[:train_num_sample, :]), (X, Y)



class BatchDataConsumer(TFBatchDataProvider):
    def __init__(self, data_x, data_y, bs=1, ih=224, iw=224, m_qs=2, nc=3, n_class=1):
        self.sm = SessionManager(bs=bs, ih=ih, iw=iw, m_qs=m_qs, nc=nc, n_class=n_class)
        self.sess = self.sm.sess

        dtn, dtt = split_into_train_test_data_set(data_x, data_y)
        self.tn_bp =  TFBatchDataProvider( self.sm, dtn[0], dtn[1], bs=bs)
        self.tt_bp = TFBatchDataProvider( self.sm, dtt[0], dtt[1], bs=bs)


        self.m = VGG('vgg15.dah5', self.sm.x, self.sm.y)

    def evaluate(self):
        MAE_T = self.m.MAE()
        t = threading.Thread(target=self.tt_bp.provider)
        t.daemon = True
        t.start()
        total_cost = 0
        for b in range(len(self.tt_bp)):
            if b == len(self.tt_bp) - 1:
                self.sm.bs.assign(self.tt_bp.last_batch_size()).eval(session=self.sess)
            cost = self.sess.run([MAE_T])
            total_cost += cost[0]
        MAE = float(total_cost)/len(self.tt_bp)
        print('testing  cost {}'.format(MAE))
        self.tt_bp.stop_providing = True
        return MAE

    def evaluate_standalone(self):
        self.m.ops_layers()
        MAE_T = self.m.MAE()
        t = threading.Thread(target=self.tt_bp.provider)
        t.daemon = True
        t.start()
        total_cost = 0
        self.sess.run(tf.initialize_all_variables())
        for b in range(len(self.tt_bp)):
            if b == len(self.tt_bp) - 1:
                self.sm.bs.assign(self.tt_bp.last_batch_size()).eval(session=self.sess)
            cost = self.sess.run([MAE_T])
            total_cost += cost[0]
        MAE = float(total_cost) / len(self.tt_bp)
        print('testing  cost {}'.format(MAE))
        self.tt_bp.stop_providing = True
        return MAE

    @staticmethod
    def get_trained_params(sess, tensors):
        params_dict = dict()
        for key in tensors:
            if isinstance(tensors[key], tf.Variable):
                params_dict[key] = sess.run(tensors[key])
        return params_dict

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
        print('testing person:', coe)
        srocc = stats.spearmanr(gt.tolist()[0], pred.tolist()[0])
        print('testing SROCC:', srocc)
        MAE = float(total_cost) / len(bp)
        return MAE

    def train(self, max_num_epoch):
        config = {
            'learning_rate':0.0001,
            'num_epoch':100,
            'MAE': 100
        }

        self.m.ops_layers()
        cost_function = self.m.MAE()
        train_op = self.m.solver()
        t = threading.Thread(target=self.tn_bp.provider)
        t.daemon = True
        t.start()
        print('training consumer ready to consume')
        self.sess.run(tf.initialize_all_variables())
        MAE_MIN = 100
        for epoch in range(max_num_epoch):
            for b in range(len(self.tn_bp)):
                if b == len(self.tn_bp) - 1:
                    self.sm.bs.assign(self.tn_bp.last_batch_size()).eval(session=self.sess)
                _, cost = self.sess.run([train_op, cost_function])
                if epoch % 5 == 0:
                    print('Training: at epoch {}, batch {}, cost {}'.format(epoch, b, cost))
            MAE = self.evaluate()

            if MAE < MAE_MIN:
                MAE_MIN = MAE
                config['num_epoch'] = epoch
                config['MAE'] = MAE
                params = self.get_trained_params(self.sess, self.m.wbp)
                H5Writer.write('trained_diq_with_min_testing_error_step2.dah5', config=config, params_dict=params)
                print("Min error================================================{}".format(MAE))

        self.tn_bp.stop_providing = True

        print('everything complete')



if __name__=="__main__":
    from diq.misc import get_xy
    x, y = get_xy()
    print(x.shape, y.shape)
    v = BatchDataConsumer(data_x=x, data_y=y, bs=7)
    v.train(200)
    #v.evaluate_standalone()os