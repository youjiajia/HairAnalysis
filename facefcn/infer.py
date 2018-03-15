import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imresize,imsave, toimage


import caffe
import caffe

caffe.set_mode_gpu()

net = caffe.Net('voc-fcn8s-atonce/deploy.prototxt', '../../datasets/FacesPartLabels/saved/snapshot-head-_iter_325000.caffemodel', caffe.TEST)

name = '4b.jpeg'
im = Image.open('../../datasets/FacesPartLabels/' + name )
im = im.resize((250, 250), Image.ANTIALIAS)
im = im.resize((512, 512), Image.ANTIALIAS)

# Turn grayscale images to 3 channels
if (im.size.__len__() == 2):
    im_gray = im
    im = Image.new("RGB", im_gray.size)
    im.paste(im_gray)

# switch to BGR and substract mean
in_ = np.array(im, dtype=np.float32)
in_ = in_[:, :, ::-1]
in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
in_ = in_.transpose((2, 0, 1))

# shape for input (data blob is N x C x H x W)
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_

# run net and take scores
net.forward()

# Compute SoftMax HeatMap
score_background = net.blobs['score_conv'].data[0][0, :, :]  # Background score
score_hair = net.blobs['score_conv'].data[0][1, :, :]  # Hair score
score_head = net.blobs['score_conv'].data[0][2, :, :]  # Head score

score_head = np.exp(score_head)
score_hair = np.exp(score_hair)
score_background = np.exp(score_background)

hmap_head = score_head / (score_head + score_hair + score_background)
hmap_hair = score_hair / (score_head + score_hair + score_background)
hmap_background = score_background / (score_head + score_hair + score_background)

out_im = np.array(im, dtype=np.float32)

out_im[:, :, 0] = hmap_hair * 255
out_im[:, :, 1] = hmap_head * 255
out_im[:, :, 2] = hmap_background * 255

imsave('../../datasets/FacesPartLabels/test_' + name , out_im)


print 'Done'