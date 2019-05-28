import sys
import multiprocessing as mp
from PIL import Image
import os, cv2
import numpy as np
from Demo.Img2Fashion.model import FPN

def initImg2FashionWorker(listenQ, sendQ, checkpoint_path, backbone_name):

    model = FPN(
        backbone_name=backbone_name,
        input_shape=(608,416,3),
        classes=14
    )
    model.load_weights(checkpoint_path)

    img2FashionPredictionCycle(model, listenQ, sendQ)

def img2FashionPredictionCycle(model, listenQ, sendQ):
    print("Img2Fashion model is ready and awaiting new jobs")
    sendQ.put("Ready")

    while (True):
        frame = listenQ.get()
        predictedSegmentation = model.predict(frame) # TODO: use model
        predictedSegmentation = np.argmax(predictedSegmentation, axis=3)[0]

        #predictedSegmentation = frame
        print(np.unique(predictedSegmentation.flatten()))

        sendQ.put(predictedSegmentation)

class Img2FashionWorker:
    def __init__(self, checkpoint_path, backbone_name):
        self._toWorkerQ   = mp.Queue(maxsize=0)
        self._fromWorkerQ = mp.Queue(maxsize=0)
        abs_check_path = os.path.abspath(checkpoint_path)
        self._workerProcess = mp.Process(target=initImg2FashionWorker,
                args=(self._toWorkerQ, self._fromWorkerQ, abs_check_path, backbone_name))

        self._workerProcess.start()
        print(self._fromWorkerQ.get())

    def predictNumpyFrame(self, frame): # Blocking call
        self._toWorkerQ.put(frame)
        return self._fromWorkerQ.get()
