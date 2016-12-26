import skimage
import skimage.io
import skimage.transform
import numpy as np
from matplotlib import pyplot as plt

synset = [l.strip() for l in open('synset.txt').readlines()]
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

def saveh5(keys, f):
    import h5py
    hfile = h5py.File('diq_dataset_random.h5', 'w')

    for i, k in enumerate(keys):
        psnr= f['psnr'][k].value
        dmos = f['dmos'][k].value
        img = f['img'][k].value
        name_1 = '{}/{}'.format('img', k)
        name_2 = '{}/{}'.format('psnr', k)
        name_3 = '{}/{}'.format('dmos', k)

        hfile.create_dataset(name=name_1, data=img)
        hfile.create_dataset(name=name_2, data=psnr)
        hfile.create_dataset(name=name_3, data=dmos)


        # quality_measure[f] = [a.psnr(f), a.get_MSE(f), a.data[i][-1]]

        # print( [a.psnr(f), a.get_MSE(f), a.data[i][-1]])
    hfile.close()


def get_xy():
    import h5py
    import numpy as np
    import random

    f = h5py.File('diq_dataset_random.h5')
    keys = list(f['psnr'].keys())

    x = np.zeros(shape=(len(keys), 224, 224, 3))
    y = np.zeros(shape=(len(keys), 1))
    for i, k in enumerate(keys):
        x[i, :, :, :] = cv2.resize(f['img'][k].value, 224, 224)
        y[i, 0] = f['psnr'][k].value
    return x, y

