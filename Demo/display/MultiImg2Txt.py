import pickle
import time
import listener_t as listener
import cv2
import multiprocessing as mp
import os
import threading

import vstream


def createOracle():
    path = os.path.dirname(os.path.realpath(__file__))
    CHECKPOINT_PATH = f"{path}/model/checkpointNew"
    VOCABULARY_PATH = f"{path}/model/word_counts.txt"

    from im2txt.im2txt.inference_utils import caption_generator
    from im2txt.im2txt.inference_utils import vocabulary
    from im2txt.im2txt import inference_wrapper
    from im2txt.im2txt import configuration

    # Init the vocabulary
    vocab = vocabulary.Vocabulary(VOCABULARY_PATH)

    # Build the graph
    import tensorflow as tf
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                                   CHECKPOINT_PATH)
    g.finalize()

    # Prepare a session object
    sess = tf.Session(graph=g)
    restore_fn(sess)  # Load the model from checkpoint.
    generator = caption_generator.CaptionGenerator(model, vocab)

    def predict(image):
        image = cv2.imencode('.jpg', image)[1].tobytes()
        for caption in generator.beam_search(sess, image):
            sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
            topSentence = " ".join(sentence)
            # Ignore begin and end words.
            return topSentence.replace('<S>', '').replace(' .', '')


    return predict


def startOracleSession(globalQueue, workerPipes):
    print("Init TF model...")
    oraclePredict = createOracle()

    while (True):
        id = globalQueue.get()
        fromWorker, toWorker = workerPipes[id]

        toWorker.put("What's your question?")
        img = fromWorker.get()
        txt = oraclePredict(img)
        print("Sending txt:", txt)
        toWorker.put(txt)


def addTxtToImage(image, text):
    h, w, _ = image.shape
    textBoxY = h - 60
    cv2.rectangle(image, (0, textBoxY), (w, h), (0, 0, 0), -1)

    white = (255, 255, 255)
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 1.5
    cv2.putText(image, text, (10, h - 10), font, font_scale, white)
    return image


class WorkerState:

    def __init__(self):
        self.currentImage = None
        self.currentText = "Thinking..."
        self._imageLock = threading.Lock()
        self._textLock = threading.Lock()

    def getRecentImage(self):
        with self._imageLock:
            img = self.currentImage.copy()
        return img

    def setRecentImage(self, img):
        with self._imageLock:
            self.currentImage = img

    def getRecentData(self):
        with self._textLock:
            return self.currentText

    def setRecentData(self, newText):
        with self._textLock:
            self.currentText = newText


def workerThread(oracleQueue, localPipes, name, sharedState):
    privateToOracle, privateFromOracle = localPipes
    while (True):
        oracleQueue.put(name)
        response = privateFromOracle.get()  # Just a random msg, telling us to send our latest img

        recentImg = sharedState.getRecentImage()
        privateToOracle.put(recentImg)

        textResponse = privateFromOracle.get()
        print("Got text response", textResponse)
        sharedState.setContent(textResponse)


def img2TxtStream(settings, globalToOracle, localPipes):
    host, port, name, url = settings

    if 'youtube' in url:
        b = vstream.YoutubeBuffer(url)
    else:
        b = vstream.VideoBuffer(url)

    b.start()
    t0 = time.time()
    next_frame = t0
    fps = 30

    sharedState = WorkerState()
    threading.Thread(target=workerThread, args=(globalToOracle, localPipes, name, sharedState)).start()

    print("Stream started:", name)
    with listener.Sender(host=host, port=int(port)) as s:
        while True:
            if len(b.buf) > 0:
                currentFrame = b.buf[0]

                sharedState.setRecentImage(currentFrame)
                recentImg = sharedState.getRecentImage()
                recentText = sharedState.getRecentData()
                textImg = addTxtToImage(recentImg, recentText)
                s.send(name, pickle.dumps(textImg))

            time.sleep(max(0, next_frame - time.time()))
            next_frame += 1 / fps

    b.stop()


def startStreams(streams):
    urls = [
        "/home/fredrik/Documents/Superbowl.mp4",
        "https://www.youtube.com/watch?v=lRa798QVEFQ"
    ]

    toOraclePipe = mp.Queue()
    workerPipes = []
    for s in streams:
        sendUrl, port, streamUrl, id = s

        fromWorkerToOracle = mp.Queue()
        fromOracleToWorker = mp.Queue()
        wPipes = (fromWorkerToOracle, fromOracleToWorker)
        workerPipes.append(wPipes)

        settings = ['localhost', port, i, urls[i]]
        mp.Process(target=img2TxtStream, args=(settings, toOraclePipe, wPipes)).start()

    startOracleSession(toOraclePipe, workerPipes)
    input("Waiting for input")


if (__name__ == '__main__'):
    startStreams()
