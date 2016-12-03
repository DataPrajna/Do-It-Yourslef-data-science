import skimage
import skimage.io
import skimage.transform
import numpy as np

synset = [l.strip() for l in open('synset.txt').readlines()]


def load_image(path):
  # load image
  img = skimage.io.imread(path)
  img = np.float32(img[:,:, ::-1])
  print(img.shape)


  #assert (0 <= img).all() and (img <= 1.0).all()
  #print "Original Image Shape: ", img.shape
  # we crop image from center
  short_edge = min(img.shape[:2])
  yy = int((img.shape[0] - short_edge) / 2)
  xx = int((img.shape[1] - short_edge) / 2)
  crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
  # resize to 224, 224
  resized_img = skimage.transform.resize(crop_img, (224, 224))
  resized_img = np.reshape(resized_img, [1, 224, 224, 3])
  return resized_img

# returns the top1 string
def print_prob(prob):
  print("prob shape", prob.shape)
  pred = np.argsort(prob)[::-1]
  print(prob[pred])

  # Get top1 label
  top1 = synset[pred[0]]
  print("Top1: {}, {}, {}".format(top1, np.argmax(prob), max(prob)))
  # Get top5 label
  top5 = [synset[pred[i]] for i in range(5)]
  print("Top5: ", top5)
  return top1