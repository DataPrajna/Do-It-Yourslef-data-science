import skimage
import skimage.io
import skimage.transform
import numpy as np
from matplotlib import pyplot as plt

#synset = [l.strip() for l in open('synset.txt').readlines()]
plt.ion()


class cv2:
  def __init__(self):
    pass

  @staticmethod
  def imread(path):
    img = skimage.io.imread(path)
    img = np.float32(img[:, :, ::-1])
    return img

  @staticmethod
  def imshow(name='doshow', img=None):
    plt.imshow(img)
    plt.show()
    plt.waitforbuttonpress(-1)
    #plt.gcf().clear()

  @staticmethod
  def resize(img, h, w):
    img = img / 255.0
    return skimage.transform.resize(img, (h, w)) * 255.0



def get_xy():
    import h5py
    import numpy as np
    f = h5py.File('diq_dataset.h5')
    keys = list(f['psnr'].keys())
    x = np.zeros(shape=(len(keys), 224, 224, 3))
    y = np.zeros(shape=(len(keys), 1))
    for i, k in enumerate(keys):
        x[i, :, :, :] = cv2.resize(f['img'][k].value, 224, 224)
        y[i, 0] = f['psnr'][k].value
    return x, y


class TIDData:
    def __init__(self, filename, pdir='/home/amishra/appspace/Do-It-Yourslef-data-science/data/TID-2008/distorted_images'):
        self.data = self.load_tid_data(filename)
        self.pdir = pdir

    def get_xy(self):
        x = np.zeros(shape=(len(self.data), 224, 224, 3))
        y = np.zeros(shape=(len(self.data), 1))
        for i in range(len(self.data)):
            x[i, :, :, :] = self.read_a_file(self.data[i][1])
            y[i, 0] = float(self.data[i][0])
        return x, y

    def read_a_file(self, fname):
        import os.path
        file_path = '{}/{}'.format(self.pdir, fname)
        if not os.path.exists(file_path):
            fname_l = list(fname)
            fname_l[0] = 'I'
            fname = ''.join(fname_l)
            file_path = '{}/{}'.format(self.pdir, fname)

        img = cv2.imread(file_path)
        return cv2.resize(img, 224, 224)

    @staticmethod
    def load_tid_data(filename):
        import csv
        data = []
        with open(filename) as fid:
            data_ptr = csv.reader(fid)
            for d in data_ptr:
                data.append(d[0].split(' '))
        return data


import os

import scipy.io as sio
import numpy
class Live:
    def __init__(self):
        self.pf = '/home/amishra/appspace/Do-It-Yourslef-data-science/data/live'
        self.num = [227, 233, 174, 174, 174]
        self.sub_dirs = ['jp2k', 'jpeg', 'wn', 'gblur', 'fastfading']
        self.data = []
        self.dmos_ref = sio.loadmat('/home/amishra/appspace/Do-It-Yourslef-data-science/data/live/dmos_realigned.mat')
        self.dmos = []
        self.create_all_img_paths()
        self.d_type = []

    def create_all_img_paths(self):
        c = 0
        for d in self.sub_dirs:
            d1 = '{}/{}'.format(self.pf, d)
            num_file = len(os.listdir(d1)) - 2
            for i in range(0, num_file):
                if self.dmos_ref['orgs'][0][c] == 0:
                    self.data.append('{}/img{}.bmp'.format(d1, i+1))
                    self.dmos.append(self.dmos_ref['dmos_new'][0][c])
                    print(c)

                c = c+1;

    def get_xy(self):
        x = np.zeros(shape=(len(self.data), 224, 224, 3))
        y = np.zeros(shape=(len(self.data), 1))
        for i in range(len(self.data)):
            x[i, :, :, :] = self.read_a_file(self.data[i])
            y[i, 0] = float(self.dmos[i])
        return x, y,

    def read_a_file(self, fname):
        img = cv2.imread(fname)
        return cv2.resize(img, 224, 224)