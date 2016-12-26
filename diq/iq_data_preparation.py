import csv
import os, sys
PARENT_DIR = "/home/amishra/workspace/prototype/IQ/"
SRC_IMG_DIR = PARENT_DIR + "src/"
sys.path.append("/home/amishra/appspace/Do-It-Yourslef-data-science/")

import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib.pylab as plt

from diq.misc import cv2
class DMOS:
    def __init__(self, data):
        self.quality = defaultdict(list)
        for i, col in enumerate(a.data):
            f = a.list_of_noisy_files[i]
            print(f)
            psnr = a.data[i][-1]
            self.quality[col[1]+'_'+col[3]].append(float(psnr))
    def plot(self):
        p = {k: np.mean(self.quality[k]) for k in self.quality}
        f = pd.DataFrame()

        names = ['AWGN', 'JPEG', 'JPEG2000', 'fnoise', 'Blur', 'contrast']

        for i in range(1, 7):
            val = []
            for j in range(1, 6):
                key = '{}_{}'.format(i, j)
                if key in p:
                    val.append(p[key])
                else:
                    val.append(0)
            print(names[i-1], val)
            f[names[i-1]] = val

        f.index = ['c_1', 'c_2', 'c_3', 'c_4', 'c_5']
        f.plot(kind='bar')
        plt.xlabel('Degree of distortion')
        plt.ylabel('DMOS')

        plt.show()
        plt.savefig('dmos_bar_plot.png')



class IQData:
    def __init__(self, filename):
        self.data = self._read_file(filename)
        self.get_source_image_name()
        self.source_image_files()
        self.distoration_image_files()

    def get_source_image_name(self):
        self.header = self.data[0]
        self.data = self.data[1:]
        self.info = dict()
        self.output = dict()

        for h in self.header:
            self.info[h] = []

        for i, v in enumerate(self.data):
            for j, h in enumerate(self.header):
                self.info[h].append(v[j])


    @staticmethod
    def _read_file(filename):
        with open(filename, 'r') as f:
            return list(csv.reader(f))


    def source_image_files(self):
        self.list_of_src_files = []
        for f in os.listdir(SRC_IMG_DIR):
            self.list_of_src_files.append(SRC_IMG_DIR+f)

    def distoration_image_files(self):
        self.list_of_noisy_files = []
        for v in self.data:
            name = PARENT_DIR+v[2]+"/"+v[0]+"."+v[2]+"."+v[3]+".png"
            self.list_of_noisy_files.append(name)

    def get_src_from_noisy_file(self, noisy_file_name):
        return SRC_IMG_DIR+noisy_file_name.split("/")[-1].split(".")[0]+".png"

    def psnr(self, filename):
        mse, max_v, img = self.get_MSE(filename)
        return 20*np.log10(max_v/mse), img

    def read_an_image(self, filename):
        return  np.float32(cv2.imread(filename))

    def get_MSE(self, filename):
        #plt.ion()

        noisy_img = np.float32(cv2.imread(filename))
        src_img = np.float32(cv2.imread(self.get_src_from_noisy_file(filename)))
        #cv2.imshow("noisy_image", np.uint8(noisy_img))
        #cv2.imshow("original image", np.uint8(src_img))
        return np.sqrt(np.mean((src_img - noisy_img) * (src_img - noisy_img))), np.max(src_img), noisy_img

    def get_samples1(self):
        step_size = 50
        hw = 28
        data = np.zeros((60000, hw, hw, 3))
        num_samples = 0;
        labels = np.zeros(60000)

        for i, f in enumerate(self.list_of_noisy_files):
            if self.info['dst_type'][i] == "JPEG":
                img = self.read_an_image(f)
                img = img / 255.0
                h, w, c = img.shape
                for y in range(0, h - hw - 1, step_size):
                    for x in range(0, w - hw - 1, step_size):
                        data[num_samples, :, :, :] = img[y:y + hw, x:x + hw, :]
                        labels[num_samples] = int(self.info['dst_lev'][i]) - 1
                        num_samples = num_samples + 1
        print(num_samples)

        randidx = np.random.randint(0, num_samples-1, num_samples)
        return data[randidx, :, :, :], labels[randidx]


    def get_samples(self):
        step_size = 5
        hw = 24
        data = np.zeros((60000, hw, hw, 3))
        num_samples = 0;
        labels = np.zeros(60000)

        for i, f in enumerate(self.list_of_noisy_files):
            if 1: #self.info['dst_type'][i] == "JPEG":
                img = self.read_an_image(f)
                img = img / 255.0
                print(f, i)
                h, w, c = img.shape
                local_patches = 0
                for y in range(0, h - hw -1 , step_size):
                    for x in range(0, w - hw-1, step_size):
                        img_sub = img[y:y+hw, x:x+hw, :]
                        std_val =  np.std(img_sub)
                        if std_val > 0.1:
                            continue
                        elif num_samples < 60000:                            #print num_samples
                            data[num_samples, :, :, :] = img_sub
                            labels[num_samples] = int(self.info['dst_lev'][i])-1
                            num_samples = num_samples + 1
                            local_patches += 1
                    if local_patches > 75:
                        break
                break
        print(num_samples)
        randidx = np.random.randint(0, num_samples - 1, num_samples)
        return data[randidx, :, :, :], labels[randidx]




def write():
    hfile = h5py.File('diq_dataset.h5', 'w')

    for i, f in enumerate(a.list_of_noisy_files):
        psnr, img = a.psnr(f)
        dmos = a.data[i][-1]
        fs = '_'.join(f.split('/'))
        name_1 = '{}/{}'.format('img', fs)
        name_2 = '{}/{}'.format('psnr', fs)
        name_3 = '{}/{}'.format('dmos', fs)

        hfile.create_dataset(name=name_1, data=img)
        hfile.create_dataset(name=name_2, data=psnr)
        hfile.create_dataset(name=name_3, data=dmos)


        # quality_measure[f] = [a.psnr(f), a.get_MSE(f), a.data[i][-1]]

        # print( [a.psnr(f), a.get_MSE(f), a.data[i][-1]])
    hfile.close()



import sys
if __name__ == "__main__":
    filename = "/home/amishra/workspace/prototype/IQ/csiq.csv"
    import h5py
    print( filename)
    a = IQData(filename)
    dmos = DMOS(a)
    dmos.plot()
    #d, l = a.get_samples()
    #print(d.shape, max(l))
    quality_measure = {}



