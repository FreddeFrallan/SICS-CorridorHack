import cv2, pickle
import sys
import Demo.Img2Fashion.utils as utils
from Demo.Img2Fashion.model import FPN
import pickle
import cv2
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import constants
import time

import keras
from keras_applications import set_keras_submodules

set_keras_submodules(
    backend=keras.backend,
    layers=keras.layers,
    models=keras.models,
    engine=keras.engine,
    utils=keras.utils,
)

BACKBONE_NAME = "resnext101"
CHECKPOINT_PATH = "data/{}-fpn-modanet.hdf5".format(BACKBONE_NAME)

def invert_image_preprocessing(x):
    x_ = np.zeros(x.shape)
    x_[:,:,:] = x
    x_[...,0] *= 0.229
    x_[...,1] *= 0.224
    x_[...,2] *= 0.225

    x_[...,0] += 0.485
    x_[...,1] += 0.456
    x_[...,2] += 0.406

    x_ *= 255

    return x_.astype(np.uint8)

def draw_text(img, seg):
    class_names = constants.get_modanet_class_names()
    for c in np.unique(seg):
        m = seg == c
        ps = np.argwhere(m)
        p = tuple(ps[len(ps)//2])
        txt = class_names[c]
        print(img.shape)
        print(seg.shape)
        print(p)
        print(txt)
        cv2.putText(img, txt, p, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))

def main():
    initialized = False
    cap = cv2.VideoCapture(2)
    cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
    fullscreen = False

    # model
    model = FPN(
        backbone_name=BACKBONE_NAME,
        input_shape=(608,416,3),
        classes=14
    )
    model.load_weights(CHECKPOINT_PATH)

    while (True):
        ret, frame = cap.read()

        t1 = time.time()
        frame = utils.prepare_cv2_frame(frame, BACKBONE_NAME)
        t2 = time.time()
        print("preprocess time: ", t2-t1)

        t1 = time.time()
        predictedSegmentation = model.predict(frame) # TODO: use model
        predictedSegmentation = np.argmax(predictedSegmentation, axis=3)[0]
        t2 = time.time()
        print("inference time: ", t2-t1)

        t1 = time.time()
        seg = utils.seg2rgb(predictedSegmentation)
        rgb = invert_image_preprocessing(frame[0])[:,:,::-1]

        print("seg: ", seg.shape)
        print("rgb: ", rgb.shape)

        alpha = 0.4
        to_show = (1.0-alpha)*seg + alpha*rgb
        to_show[predictedSegmentation == 0, :] = rgb[predictedSegmentation == 0, :]
        to_show = to_show.astype(np.uint8)

        #draw_text(to_show, predictedSegmentation)

        if fullscreen:
            cv2.setWindowProperty("frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.imshow("frame", to_show)
        else:
            cv2.imshow("frame", to_show)
        t2 = time.time()
        print("drawing time: ", t2-t1)

        key=cv2.waitKey(1) & 0xff
        print(key)

        if (key == ord('q')):
            break
        if (key == ord('f')):
            fullscreen = not fullscreen
            print("fullscreen: ", fullscreen)

    # CameraStream cleanup
    cap.release()
    cv2.destroyAllWindows()

if (__name__ == '__main__'):
    main()
