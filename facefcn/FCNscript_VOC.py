import numpy as np
import cv2
import caffe


# Set caffe to use GPU and load the FCN net
caffe.set_mode_gpu()
net = caffe.Net('voc-fcn8s-atonce/deploy.prototxt', '../../datasets/FacesPartLabels/snapshots/fcn8s-atonce-pascal.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W)
net.blobs['data'].reshape(1, 3, 512, 512)

# initialize video camera input
cap = cv2.VideoCapture(0)

SafeVideo = True

if (SafeVideo):
    video = cv2.VideoWriter('output.avi', 0, 12.0, (2*640, 480))
    video2 = cv2.VideoWriter('output2.avi', 0, 12.0, (2*640, 480))


while (True):

    # Capture frame-by-frame
    ret, frame = cap.read()

    # resize to 512x512
    frame = cv2.resize(frame, (150, 150))
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
    scores = np.exp(net.blobs['score_conv'].data[0][0, :, :])
    for i in range(1,20):
        scores = scores + np.exp(net.blobs['score_conv'].data[0][i, :, :])

    hmap_person = np.exp(net.blobs['score_conv'].data[0][15, :, :]) / scores
    # Display the input frame and output heatmap
    frame = cv2.resize(frame, (640, 480))
    #cv2.imshow('frame', cv2.flip(frame, 1))
    cv2.imshow('frame', frame)
    out = np.array((hmap_person * 255), dtype=np.uint8)
    out = cv2.applyColorMap(out, cv2.COLORMAP_JET)
    out = cv2.resize(out, (640, 480))
    cv2.imshow('person', out)

    hmap_person = np.exp(net.blobs['score_conv'].data[0][5, :, :]) / scores
    # Display the input frame and output heatmap
    frame = cv2.resize(frame, (640, 480))
    #cv2.imshow('frame', cv2.flip(frame, 1))
    cv2.imshow('frame', frame)
    out = np.array((hmap_person * 255), dtype=np.uint8)
    out = cv2.applyColorMap(out, cv2.COLORMAP_JET)
    out = cv2.resize(out, (640, 480))
    cv2.imshow('bottle', out)

    hmap_person = np.exp(net.blobs['score_conv'].data[0][16, :, :]) / scores
    # Display the input frame and output heatmap
    frame = cv2.resize(frame, (640, 480))
    #cv2.imshow('frame', cv2.flip(frame, 1))
    cv2.imshow('frame', frame)
    out = np.array((hmap_person * 255), dtype=np.uint8)
    out = cv2.applyColorMap(out, cv2.COLORMAP_JET)
    out = cv2.resize(out, (640, 480))
    cv2.imshow('plant', out)

    hmap_person = np.exp(net.blobs['score_conv'].data[0][9, :, :]) / scores
    # Display the input frame and output heatmap
    frame = cv2.resize(frame, (640, 480))
    #cv2.imshow('frame', cv2.flip(frame, 1))
    cv2.imshow('frame', frame)
    out = np.array((hmap_person * 255), dtype=np.uint8)
    out = cv2.applyColorMap(out, cv2.COLORMAP_JET)
    out = cv2.resize(out, (640, 480))
    cv2.imshow('chair', out)


    if (SafeVideo):
        I = np.concatenate((frame, out), axis=1)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
if (SafeVideo):
    video.release()
cv2.destroyAllWindows()