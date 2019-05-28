import sys
import Network.Producer.Node as Producer
import Network.Consumer.ServerNode as Consumer
import Demo.Img2Fashion.ModelWorkerMRCNN as Img2Fashion
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

def main():
    print("starting consumer node ...")
    consumerNode = Consumer.Node(1234)
    print("starting producer node ...")
    producerNode = Producer.Node(1235)
    print("starting worker ...")
    worker = Img2Fashion.Img2FashionWorker()

    while (True):
        print("waiting for update ...")
        pickledFrame = consumerNode.getUpdate()
        frame = pickle.loads(pickledFrame)[0]

        # BGR -> RFGB
        frame = frame[:,:,::-1]

        print("predicting frame ...")
        r = worker.predictNumpyFrame(frame)
        producerNode.updateContent(pickle.dumps(r))

if (__name__ == '__main__'):
    main()
