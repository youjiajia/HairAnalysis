import caffe
import numpy as np
from PIL import Image, ImageOps
import random
import time

class FacesPartLabelsDataLayer(caffe.Layer):
    def setup(self, bottom, top):

        params = eval(self.param_str)
        self.data_dir = params['FacesPartLabels_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        self.crop_size = 500

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.data_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.mirror = random.randint(0, 1)
        self.scaling_factor = random.uniform(1,1.4)
        self.padding = random.randint(0, 250)
        self.padding_prob = random.randint(0, 1)
        width, height = (int((600 + self.padding) * self.scaling_factor), (int((600 + self.padding) * self.scaling_factor)))

        self.left = random.randint(0, width - self.crop_size - 1)
        self.top_pos = random.randint(0, height - self.crop_size - 1)

        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])

        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)



    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        idx = idx.split()

        im = Image.open(self.data_dir + '/lfw_funneled/' + idx[0] + '/' + idx[0] + '_' + idx[1].zfill(4) + '.jpg')
        im = im.resize((600, 600), Image.ANTIALIAS)
        if( im.size.__len__() == 2):
            im_gray = im
            im = Image.new("RGB", im_gray.size)
            im.paste(im_gray)

        if self.mirror == 1:
            im = ImageOps.mirror(im)
        if self.padding_prob > 0.5:
            im = self.add_padding(im)
        im = self.rescale_image(im)
        im = self.random_crop(im)

        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        idx = idx.split()
        im = Image.open(self.data_dir + '/parts_lfw_funneled_gt_images/' + idx[0] + '_' + idx[1].zfill(4) + '.ppm')
        im = im.resize((600, 600), Image.ANTIALIAS)

        if self.mirror == 1:
            im = ImageOps.mirror(im)
        if self.padding_prob > 0.5:
            im = self.add_padding(im)
        im = self.rescale_image(im)
        im = self.random_crop(im)


        label = np.array(im, dtype=np.uint8)
        out_label = np.zeros((self.crop_size,self.crop_size), dtype=np.uint8)
        out_label[label[:, :, 0] > 0] = 1
        out_label[label[:, :, 1] > 0] = 2
        out_label = out_label[np.newaxis, ...]
        return out_label


    def rescale_image(self, im):
        width, height = im.size
        im = im.resize((int(width * self.scaling_factor), int(height * self.scaling_factor)), Image.ANTIALIAS)
        return im

    def random_crop(self, im):
        im = im.crop((self.left, self.top_pos, self.left + self.crop_size, self.top_pos + self.crop_size))
        return im

    def add_padding(self, im):
        im = np.array(im, dtype=np.uint8)
        im = np.pad(im,((self.padding,self.padding),(self.padding,self.padding),(0,0)), mode = 'constant')
        return Image.fromarray(im,'RGB')