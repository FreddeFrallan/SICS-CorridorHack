import sys
import multiprocessing as mp
from PIL import Image
import os, cv2
import numpy as np

# Mask RCNN
ROOT_DIR    = "/home/john/gits/Mask_RCNN/"
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

import tensorflow as tf
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
from samples.modanet import modanet

def initImg2FashionWorker(listenQ, sendQ):
    config = modanet.ModanetConfig()
    # Override the training configurations with a few
    # changes for inferencing.
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    DEVICE = "/gpu:1"  # /cpu:0 or /gpu:0
    TEST_MODE = "inference"

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    weights_path = model.find_last()

    model.load_weights(weights_path, by_name=True)

    img2FashionPredictionCycle(model, listenQ, sendQ)

def img2FashionPredictionCycle(model, listenQ, sendQ):
    print("Img2Fashion model is ready and awaiting new jobs")
    sendQ.put("Ready")

    while (True):
        frame = listenQ.get()
        results = model.detect([frame], verbose=1)
        print("prediction made ... ")
        sendQ.put(results[0])

class Img2FashionWorker:
    def __init__(self):
        self._toWorkerQ   = mp.Queue(maxsize=0)
        self._fromWorkerQ = mp.Queue(maxsize=0)
        self._workerProcess = mp.Process(target=initImg2FashionWorker,
                args=(self._toWorkerQ, self._fromWorkerQ))

        self._workerProcess.start()
        print(self._fromWorkerQ.get())

    def predictNumpyFrame(self, frame): # Blocking call
        self._toWorkerQ.put(frame)
        return self._fromWorkerQ.get()
