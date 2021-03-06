############################### Info ###################################
"""
Author: Muhammad Umar Riaz
Date Updated: June 2017
"""
############################### Import #################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn import preprocessing, svm
from skimage import io, color, segmentation
from math import ceil
from matting import closed_form_matting
import os
import sys
import cv2
import glob
import time
# import matplotlib.pyplot as plt
import numpy as np
import skimage
import caffe
import argparse
from scipy.misc import imresize,imsave, toimage

############################### Parser #################################

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input-dir', default='./Data/' ,help='Directory containing images (jpg format) to process')
parser.add_argument('-o','--output-dir', default='./Output/' ,help='Output directory')
parser.add_argument('--caffeNet-dir', default='./Tools/CaffeNet/', help='Directory containing caffeNet model')
parser.add_argument('--gpu-mode', default=False, help='Use gpu mode for caffe')
parser.add_argument('--training-features-dir', default='./Features/', help='Directory containing training features file')
parser.add_argument('--visual-output', default=False, help='Visual output of the processing (True for visualing, False otherwise)')
parser.add_argument('--visual-output-save', default=False, help='Save the visual output of the processing (True for saving, False otherwise)')
args = parser.parse_args()

############################ Functions #################################

def sliding_window(image, stepSize):
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            yield (x, y)

def load_image(filename, color=True):
    return skimage.img_as_float(skimage.io.imread(filename)).astype(np.float32)

def compute_uPattern():
    uPattern = []
    for n in range(256):
        nbin = bin(n)
        nstr = str(nbin)
        nstr = nstr[2:]
        byte = np.zeros(shape = [8])
        for x in range (8-len(nstr),8,1):
            byte[x] = nstr[x - 8 + len(nstr)]
        step = 0
        for bit in range(7):
            if byte[bit] != byte[bit+1]:
                step += 1
        if step < 3 :
            uPattern.append(n)
        uPattern_values = np.asarray(uPattern)
    return uPattern_values

def LTP_feature_extraction(image_block, reorder_vector, t, exp, uPattern_values):
    block_hist_upper = np.zeros(shape=[256])
    block_hist_lower = np.zeros(shape=[256])
    block_hist_uPattern_upper = np.zeros(shape=[59])
    block_hist_uPattern_lower = np.zeros(shape=[59])
    brows, bcols = image_block.shape
    # For each pixel in our image, ignoring the borders...
    for brow in range(1,brows-1):
        for bcol in range(1,bcols-1):
            # Get centre
            center = image_block[brow,bcol]
            # Get neighbourhood
            pixels = image_block[brow-1 : brow+2, bcol-1 : bcol+2]
            pixels = np.ravel(pixels)
            neighborhood = pixels[reorder_vector]
            # Get ranges and determine LTP
            low = center - t
            high = center + t
            block_LTP_out = np.zeros(neighborhood.shape)
            block_LTP_out[neighborhood < low] = -1
            block_LTP_out[neighborhood > high] = 1
            # Get upper and lower patterns -> LBP
            upper = np.copy(block_LTP_out)
            upper[upper == -1] = 0
            du = np.sum( pow(2, exp) * upper )
            lower = np.copy(block_LTP_out)
            lower[lower == 1] = 0
            lower[lower == -1] = 1
            dl = np.sum( pow(2, exp) * lower )
            if any(uPattern_values == du):
                block_hist_uPattern_upper[uPattern_values == du] += 1
            else:
                block_hist_uPattern_upper[58] += 1

            if any(uPattern_values == dl):
                block_hist_uPattern_lower[uPattern_values == dl] += 1
            else:
                block_hist_uPattern_lower[58] += 1
    fileRow = np.concatenate([block_hist_uPattern_lower, block_hist_uPattern_upper])
    return fileRow

###########################  Data Loading  #############################

org_images          = sorted(glob.glob(args.input_dir+"*.jpg"))
my_deploy_prototext = args.caffeNet_dir + 'deploy.prototxt'
my_caffemodel       = args.caffeNet_dir + 'model_caffenet.caffemodel'
my_meanfile         = args.caffeNet_dir + 'mean.npy'
feature_array       = np.load(args.training_features_dir + 'trainfeature.npy')
label_array         = np.load(args.training_features_dir + 'trainlabel.npy')
detect_step         = 33
seg_step            = 3
patch_dim           = 231
hair_thr            = 65
nonhair_thr         = 15

if args.gpu_mode:
    caffe.set_mode_gpu()
    caffe.set_device(0)
else:
    caffe.set_mode_cpu()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

####################### CaffeNet Initialization ########################

net = caffe.Net(my_deploy_prototext, my_caffemodel, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load(my_meanfile).mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)
net.blobs['data'].reshape(1,3,227,227)

################## Detection: Classifier Training ######################

print "--> Classifier training"

imp                 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 1)
feature_array       = imp.fit_transform(feature_array)
scalerDet           = preprocessing.StandardScaler().fit(feature_array)
feature_array       = scalerDet.transform(feature_array)
hair_idx            = np.where(label_array == 1.0)[0]
nonhair_idx         = np.where(label_array == 0.0)[0]
hair_samples_no     = hair_idx.shape[0]
nonhair_samples_no  = nonhair_idx.shape[0]

print "Hair samples: ", hair_samples_no, " NonHair samples: ", nonhair_samples_no

hair_array          = [feature_array[i] for i in hair_idx]
nonhair_array       = [feature_array[i] for i in nonhair_idx]
hair_label          = [1] * hair_samples_no
nonhair_label       = [0] * nonhair_samples_no
train_feature       = np.asarray(hair_array + nonhair_array)
train_label         = np.asarray(hair_label + nonhair_label)
my_clf_det          = RandomForestClassifier(n_estimators = 100)
my_clf_det          = my_clf_det.fit(train_feature, train_label)

################### Detection: Classifier Testing #######################

for n in range(len(org_images)):

    print "Processing: ", org_images[n].split('/')[-1:][0][:-4]

    org_image_rgb        = load_image(org_images[n])
    org_image            = color.rgb2gray(org_image_rgb)
    border_org_image_rgb = cv2.copyMakeBorder(org_image_rgb, patch_dim, patch_dim, patch_dim, patch_dim, cv2.BORDER_CONSTANT,value=cv2.mean(org_image_rgb)[:3])
    border_org_image     = cv2.copyMakeBorder(org_image, patch_dim, patch_dim, patch_dim, patch_dim, cv2.BORDER_CONSTANT,value=cv2.mean(org_image)[0])
    
    if args.visual_output_save or args.visual_output:
        image_footprint  = border_org_image_rgb.copy()
    output_image         = np.zeros(border_org_image_rgb.shape[:2], dtype = int)

    print "Hair Detection at patch-level"

    # tic = time.clock()
    tic = time.time()
    num = 0
    for (x, y) in sliding_window(border_org_image, stepSize = detect_step):
        num += 1
        print time.time()
        if (y + patch_dim > border_org_image.shape[0]) or (x + patch_dim > border_org_image.shape[1]):
            continue
        image_block_rgb             = border_org_image_rgb[ y:y+patch_dim , x:x+patch_dim, :]
        image_block                 = border_org_image[ y:y+patch_dim , x:x+patch_dim]
        image_block_rgb_227         = image_block_rgb[2:229,2:229,:]
        net.blobs["data"].data[...] = transformer.preprocess("data", image_block_rgb_227)
        out                         = net.forward()
        fVector                     = net.blobs['fc7'].data[0].copy()
        feature_array               = imp.transform(fVector.reshape(1,-1))
        feature_array               = scalerDet.transform(feature_array)
        hair_prediction             = my_clf_det.predict(feature_array)
        if hair_prediction[0] == 1.0:
            output_image[ y:y+patch_dim, x:x+patch_dim ] += 1
            if args.visual_output_save or args.visual_output:
                image_footprint[y:y+patch_dim, x:x+patch_dim, :] -= 0.04
                hair_color = (0,0,255)
        elif hair_prediction[0] == 0.0:
            if args.visual_output_save or args.visual_output:
                hair_color = (0,255,0)

        if args.visual_output:
            clone = image_footprint.copy()
            cv2.rectangle(clone, (x, y ), (x+patch_dim, y+patch_dim), hair_color, 2)
            clone = clone[patch_dim:org_image.shape[0], patch_dim:org_image.shape[1]]
            cv2.imshow("Window", np.fliplr(clone.reshape(-1,3)).reshape(clone.shape))
            cv2.waitKey(1)
            time.sleep(0.025)
    # toc = time.clock()
    toc = time.time()

    print "Hair Detection completed"
    print "Processing time: ", toc-tic, "seconds"

    output_image           = output_image [patch_dim : patch_dim + org_image.shape[0], patch_dim : patch_dim + org_image.shape[1]]
    unique_val, counts_val = np.unique(output_image, return_counts=True)
    unique_counts_val      = dict(zip(unique_val, counts_val))

    if unique_counts_val.get(0) == output_image.shape[0] * output_image.shape[1] or output_image.max() <= 5:
        print "Processing finished. All pixels in the input image are labelled as nonhair."
        cv2.imwrite(args.output_dir + "Hair-region.png", output_image.astype(np.int))
        continue
    elif unique_counts_val.get(49) == output_image.shape[0] * output_image.shape[1]:
        print "Processing finished. All pixels in the input image are labelled as hair."
        output_image[output_image == 49] = 255
        cv2.imwrite(args.output_dir + "Hair-region.png", output_image.astype(np.int))
        continue

    hair_thr_relative    = int(ceil(float( output_image.max() * hair_thr ) / 100))
    nonhair_thr_relative = int(ceil(float( output_image.max() * nonhair_thr ) / 100))
    Hair_region          = output_image >= hair_thr_relative
    NonHair_region       = output_image <= nonhair_thr_relative
    getborder = []
    cv2.imwrite(args.output_dir + "Hair-region.png", Hair_region.astype(np.int)*255)
    cv2.imwrite(args.output_dir + "NonHair-region.png", NonHair_region.astype(np.int)*255)



    im = cv2.imread("./Data/Example.jpg")
    mask = np.zeros(im.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (0,0,im.shape[0],im.shape[1])

    for x in xrange(im.shape[0]):
        for y in xrange(im.shape[1]):
            if Hair_region[x][y]:
                mask[x][y] = 1
            elif NonHair_region[x][y]:
                mask[x][y] = 0
            else:
                mask[x][y] = 3

    cv2.grabCut(im,mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_MASK)
    mask2 = np.where((mask==2)|(mask==0)|(mask==3),0,1).astype('uint8')

    im = im*mask2[:,:,np.newaxis]
    print 'Done'
    toc = time.time()

    print "Processing time: ", toc-tic, "seconds"

    imsave('./Output/test_hair.jpg' , im)



    # for (x, y) in sliding_window(Hair_region, stepSize = detect_step):
    #     value = Hair_region[y][x]
    #     if value and  ((y-detect_step > 0 and Hair_region[y-detect_step][x] != value) or (x-detect_step > 0 and Hair_region[y][x-detect_step] != value) or (y+detect_step < Hair_region.shape[0] and Hair_region[y+detect_step][x] != value) or (x+detect_step < Hair_region.shape[1] and Hair_region[y][x+detect_step] != value)):
    #         getborder.append((x,y))
    # for (x,y) in getborder:
    #     for i in xrange(detect_step):
    #         for i2 in xrange(detect_step):
    #             if y-i >= 0 and x-i2 > 0:
    #                 Hair_region[y+i][x+i2] = False
    #             if x-i >= 0 and y-i2 >0:
    #                 Hair_region[y+i2][x+i] = False
    # left = NonHair_region.shape[1]
    # right = 0
    # for (x, y) in sliding_window(NonHair_region, stepSize = detect_step):
    #     value = NonHair_region[y][x]
    #     if value and  ((x-detect_step > 0 and NonHair_region[y][x-detect_step] != value) or (x+detect_step < NonHair_region.shape[1] and NonHair_region[y][x+detect_step] != value) or (y-detect_step > 0 and NonHair_region[y-detect_step][x] != value) or (y+detect_step < NonHair_region.shape[0] and NonHair_region[y+detect_step][x] != value)):
    #         if left > x and x >= 0:
    #             left = x
    #         if right <= x + detect_step and x + detect_step < NonHair_region.shape[1] - detect_step:
    #             right = x + detect_step
    # print left
    # print right
    # for x in xrange(NonHair_region.shape[1]):
    #     for y in xrange(NonHair_region.shape[0]):
    #         if x>= left and x <= right:
    #             NonHair_region[y][x] = False
    #         else:
    #             NonHair_region[y][x] = True
    # cv2.imwrite(args.output_dir + "Hair-region2.png", Hair_region.astype(np.int)*255)
    # cv2.imwrite(args.output_dir + "NonHair-region2.png", NonHair_region.astype(np.int)*255)

    # alpha = closed_form_matting.closed_form_matting_with_trimap(org_image_rgb, NonHair_region.astype(np.int)*255);
    # cv2.imwrite(args.output_dir + "testalpha.png", alpha * 255.0)
    # cv2.imwrite(args.output_dir + "testalpha2.png", output_image * 255.0)
    # newimg=np.zeros([NonHair_region.shape[0],NonHair_region.shape[1],3],np.uint8)
    # for x in xrange(NonHair_region.shape[1]):
    #     for y in xrange(NonHair_region.shape[0]):
    #         if x>= left and x <= right:
    #             if Hair_region[y][x]:
    #                 newimg[:][y][x] = 255
    #             else:
    #                 newimg[:][y][x] = 128
    #         # elif (x < detect_step and y < detect_step) or (y >= NonHair_region.shape[0]-detect_step and x >= NonHair_region.shape[1]-detect_step) or (y >= NonHair_region.shape[0]-detect_step and x < detect_step) or (y < detect_step and x >= NonHair_region.shape[1]-detect_step):
    #         #     newimg[:][y][x] = 0
    #         # else:
    #         #     newimg[:][y][x] = 128
    # cv2.imwrite(args.output_dir + "finall.png", newimg)
    # image = cv2.imread("./Data/Example.jpg", cv2.IMREAD_COLOR) / 255.0
    # trimap = cv2.imread(args.output_dir + "finall.png", cv2.IMREAD_GRAYSCALE) / 255.0
    # output = closed_form_matting.closed_form_matting_with_trimap(image, trimap)
    # cv2.imwrite(args.output_dir + "finall2.png", output * 255.0)
