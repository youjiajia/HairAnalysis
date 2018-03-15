import numpy as np
import cv2
import caffe


# Set caffe to use GPU and load the FCN net
caffe.set_mode_gpu()
net = caffe.Net('voc-fcn8s-atonce/deploy.prototxt', '../../datasets/FacesPartLabels/saved/snapshot-head-_iter_325000.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W)
net.blobs['data'].reshape(1, 3, 512, 512)

# initialize video camera input
cap = cv2.VideoCapture(0)

SafeVideo = True

if (SafeVideo):
    video = cv2.VideoWriter('output.avi', 0, 12.0, (640, 480))
    # video2 = cv2.VideoWriter('output2.avi', 0, 12.0, (2*640, 480))


while (True):

    # Capture frame-by-frame
    ret, frame = cap.read()

    # resize to 512x512
    frame = cv2.resize(frame, (250, 250))
    frame = cv2.resize(frame, (512, 512))


    # switch to BGR, subtract mean, and make dims C x H x W for Caffe
    in_ = np.array(frame, dtype=np.float32)
    # in_ = in_[:, :, ::-1]
    in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
    in_ = in_.transpose((2, 0, 1))

    # set input data+
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()

    # softmax net outputs
    score_background = net.blobs['score_conv'].data[0][0, :, :]   #Background score
    score_hair = net.blobs['score_conv'].data[0][1, :, :]   #Hair score
    score_head = net.blobs['score_conv'].data[0][2, :, :]   #Head score

    score_head = np.exp(score_head)
    score_hair = np.exp(score_hair)
    score_background = np.exp(score_background)

    hmap_head = score_head / (score_head + score_hair + score_background)
    hmap_hair = score_hair / (score_head + score_hair + score_background)
    hmap_background = score_background / (score_head + score_hair + score_background)

    # Display the input frame and output heatmap
    frame = cv2.resize(frame, (640, 480))
    #cv2.imshow('frame', cv2.flip(frame, 1))
    #cv2.imshow('frame', frame)

    # out = np.array((hmap_head * 255), dtype=np.uint8)
    # out = cv2.applyColorMap(out, cv2.COLORMAP_JET)
    # out = cv2.resize(out, (640, 480))
    # cv2.namedWindow("hmap_head")
    # cv2.imshow('hmap_head', out)

    out_im = np.array(cv2.resize(frame, (512, 512)))
    out_im[:,:,0] = hmap_background * 0
    out_im[:,:,1] = hmap_head * 255
    out_im[:,:,2] = hmap_hair * 255
    out_im = cv2.resize(out_im, (640, 480))
    cv2.namedWindow("combined")
    cv2.imshow('combined', out_im)

    out_im = np.array(cv2.resize(frame, (512, 512)))
    out_im[:,:,0] = hmap_background * 0
    out_im[:,:,1] = hmap_head * 255
    out_im[:,:,2] = hmap_hair * 255
    out_im = cv2.resize(out_im, (640, 480))
    out_both = cv2.addWeighted(frame, 0.6, out_im, 0.4, 0)
    cv2.namedWindow("out_both")
    cv2.imshow('out_both', out_both)


    if (SafeVideo):
        # I = np.concatenate((frame, out_both), axis=1)
        video.write(out_both)
        # video2.write(out_both)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
if (SafeVideo):
    video.release()
cv2.destroyAllWindows()