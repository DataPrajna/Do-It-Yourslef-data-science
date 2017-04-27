import skimage
import skimage.io
import skimage.transform
import numpy as np

synset = [l.strip() for l in open('synset.txt').readlines()]


def load_image(path):
  img = skimage.io.imread(path)
  VGG_MEAN = [128, 128, 128]
  img = np.float32(img[:,:, ::-1]) - VGG_MEAN
  print(np.max(img), np.min(img))
  short_edge = min(img.shape[:2])
  yy = int((img.shape[0] - short_edge) / 2)
  xx = int((img.shape[1] - short_edge) / 2)
  crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
  resized_img = skimage.transform.resize(crop_img, (224, 224))
  resized_img = np.reshape(resized_img, [1, 224, 224, 3])
  return resized_img + VGG_MEAN

def print_prob(prob):
  pred = np.argsort(prob)[::-1]
  top1 = synset[pred[0]]
  print("Top1: {}, {}, {}".format(top1, np.argmax(prob), max(prob)))
  top5 = [synset[pred[i]] for i in range(5)]
  print("Top5: ", top5)
  return top1