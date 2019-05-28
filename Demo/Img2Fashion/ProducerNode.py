import sys
import Network.Producer.Node as Producer
import Network.Consumer.Node as Consumer
import Demo.Img2Fashion.ModelWorker as Img2Fashion
import Demo.Img2Fashion.utils as utils
import pickle
import cv2
import numpy as np

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

def main():
    print("starting consumer node ...")
    consumerNode = Consumer.Node('localhost', 1234)
    print("starting producer node ...")
    producerNode = Producer.Node(1235)
    print("starting worker ...")
    worker = Img2Fashion.Img2FashionWorker(CHECKPOINT_PATH, BACKBONE_NAME)

    while (True):
        print("waiting for update ...")
        pickledFrame = consumerNode.getUpdate()
        frame = pickle.loads(pickledFrame)[0]

        frame = utils.prepare_cv2_frame(frame, BACKBONE_NAME)

        print("predicting frame ...")
        predictedSegmentation = worker.predictNumpyFrame(frame)
        predictedSegmentation = np.expand_dims(predictedSegmentation, axis=2)
        seg_and_rgb = np.concatenate((predictedSegmentation, frame[0]), axis=2)
        producerNode.updateContent(pickle.dumps(seg_and_rgb))

if (__name__ == '__main__'):
    main()
