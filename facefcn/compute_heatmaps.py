import sys
sys.path.append('/home/imatge/caffe-master/python')
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import caffe
import surgery, score
from scipy.misc import imresize, imsave, toimage
import time
import cv2
import os


# Run in GPU
caffe.set_device(0)
caffe.set_mode_gpu()

#Compute heatmaps from images in txt
# test = open('../../datasets/FacesPartLabels/parts_test.txt').read().splitlines()
test = open('../../datasets/INDO/indices.txt').read().splitlines()

# load net
net = caffe.Net('voc-fcn8s-atonce/deploy.prototxt', '../../datasets/FacesPartLabels/saved/snapshot-head-_iter_325000.caffemodel', caffe.TEST)


print 'Computing heatmaps ...'

count = 0
start = time.time()

for idx in test:

    count = count + 1
    if count % 100 == 0:
        print count

    # load image
    idx = idx.split()
    # im = Image.open('../../datasets/FacesPartLabels/lfw_funneled/' + idx[0] + '/' + idx[0] + '_' + idx[1].zfill(4) + '.jpg')
    im = Image.open('../../datasets/INDO/img/' + idx[1] + '/images/CAM_' + idx[2] + '.bmp')
    im = im.resize((250, 250), Image.ANTIALIAS)
    im = im.resize((512, 512), Image.ANTIALIAS)

    # Turn grayscale images to 3 channels
    if (im.size.__len__() == 2):
        im_gray = im
        im = Image.new("RGB", im_gray.size)
        im.paste(im_gray)

    #switch to BGR and substract mean
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))

    # shape for input (data blob is N x C x H x W)
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_

    # run net and take scores
    net.forward()

    # Compute SoftMax HeatMap
    score_background = net.blobs['score_conv'].data[0][0, :, :]   #Background score
    score_hair = net.blobs['score_conv'].data[0][1, :, :]   #Hair score
    score_head = net.blobs['score_conv'].data[0][2, :, :]   #Head score

    score_head = np.exp(score_head)
    score_hair = np.exp(score_hair)
    score_background = np.exp(score_background)

    hmap_head = score_head / (score_head + score_hair + score_background)
    hmap_hair = score_hair / (score_head + score_hair + score_background)
    hmap_background = score_background / (score_head + score_hair + score_background)

    out_im = np.array(im, dtype=np.float32)

    out_im[:,:,0] = hmap_hair * 255
    out_im[:,:,1] = hmap_head * 255
    out_im[:,:,2] = hmap_background * 255


    #Save CSV heatmap
    # pixels = np.asarray(hmap_softmax)
    # np.savetxt('/home/imatge/caffe-master/data/coco-text/csv_heatmaps/voc-fcn8s-atonce-104000it/' + idx + '.csv', pixels, delimiter=",")

    #Save PNG softmax heatmap
    # hmap_softmax_2save = (255.0 * hmap_softmax).astype(np.uint8)
    # hmap_softmax_2save = Image.fromarray(hmap_softmax_2save)
    # hmap_softmax_2save.save('/home/imatge/caffe-master/data/coco-text/heatmaps-withoutIllegible/' + idx + '.png')


    # Save color softmax heatmap
    # fig = plt.figure(frameon=False)
    # fig.set_size_inches(5.12,5.12)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    # ax.imshow(hmap_softmax, aspect='auto', cmap="jet")
    # fig.savefig('/home/imatge/caffe-master/data/coco-text/heatmaps/' + idx + '-ht.jpg')
    # plt.close(fig)

    # Save combined color heatmap
    # imsave('../../datasets/FacesPartLabels/heatmaps/' + idx[0] + '_' + idx[1].zfill(4) + '.jpg', out_im)
    if not os.path.exists('../../datasets/INDO/heatmaps3/' + idx[1] + '/'):
        os.makedirs('../../datasets/INDO/heatmaps3/' + idx[1] + '/')
    imsave('../../datasets/INDO/heatmaps3/' + idx[1] + '/' + idx[2] + '.jpg', out_im)


    print 'Heatmap saved for image: ' + str(idx[0])

end = time.time()
print 'Total time elapsed in heatmap computations'
print(end - start)
print 'Time per image'
print(end - start)/test.__len__()