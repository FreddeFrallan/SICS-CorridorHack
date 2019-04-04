import multiprocessing
import sys
import listener_t as listener
import vstream
import time
import pickle
import cv2

import Demo.Img2Txt.ModelWorker as Img2Txt

CHECKPOINT_PATH = "model/checkpointNew"
VOCABULARY_PATH = "model/word_counts.txt"

def worker(queue_img, queue_txt):
    queue_txt.put('Thinking...')
    

    worker = Img2Txt.Img2TextWorker(CHECKPOINT_PATH, VOCABULARY_PATH) 

    while True:
        img = queue_img.get()
        print(f'got img {img.shape}')
        predictedSentences = worker.predictNumpyFrame(img)
        topSentence = predictedSentences[0]
        text = topSentence
        text = text.replace('<S>', '').replace(' .', '')
        queue_txt.put(text)

######################################################################
        
queue_img = multiprocessing.Queue()
queue_txt = multiprocessing.Queue()
multiprocessing.Process(target=worker, args=(queue_img, queue_txt,)).start()

def annotate(img, text):
    if not queue_txt.empty():
        text = queue_txt.get()
        queue_img.put(img)
        
    h, w, _ = img.shape
    white = (255, 255, 255)
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 1
    cv2.putText(img, text, (10, h-10 ), font, font_scale, white ) 
    return text
