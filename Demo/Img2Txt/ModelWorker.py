import multiprocessing as mp
import os, cv2


def initImg2TxtWorker(listenQ, sendQ, checkpointPath, vocabularyPath):
    from Demo.Img2Txt.im2txt.im2txt.inference_utils import caption_generator
    from Demo.Img2Txt.im2txt.im2txt.inference_utils import vocabulary
    from Demo.Img2Txt.im2txt.im2txt import inference_wrapper
    from Demo.Img2Txt.im2txt.im2txt import configuration

    import tensorflow as tf
    print("Starting Img2Txt model...")
    print("Using checkpoint:", checkpointPath)
    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(), checkpointPath)
    g.finalize()

    vocab = vocabulary.Vocabulary(vocabularyPath)  # Init the vocabulary
    with tf.Session(graph=g) as sess:
        restore_fn(sess)  # Load the model from checkpoint.

        # Prepare the caption generator. Here we are implicitly using the default
        # beam search parameters. See caption_generator.py for a description of the
        # available beam search parameters.
        generator = caption_generator.CaptionGenerator(model, vocab)
        img2TxtPredictionCycle(sess, generator, vocab, listenQ, sendQ)


def img2TxtPredictionCycle(sess, generator, vocab, listenQ, sendQ):
    print("Img2Txt model is ready and awaiting new jobs")
    sendQ.put("Ready")
    while (True):
        image = listenQ.get()  # Wait for a new job from the spawning process

        topSentences = []
        captions = generator.beam_search(sess, image)
        for i, caption in enumerate(captions):
            # Ignore begin and end words.
            sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
            sentence = " ".join(sentence)

            topSentences.append(sentence)

        sendQ.put(topSentences)


class Img2TextWorker:
    '''
    Simple class for interfacing with the Img2Txt prediction model.
    Spawns a new process that runs the model on the first GPU.

    TODO: Options concerning GPU settings.
    '''

    def __init__(self, checkpointPath, vocabularyPath):
        self._toWorkerQ = mp.Queue(maxsize=0)
        self._fromWorkerQ = mp.Queue(maxsize=0)
        absCheckPath = os.path.abspath(checkpointPath)
        absVocabPath = os.path.abspath(vocabularyPath)
        self._workerProc = mp.Process(target=initImg2TxtWorker,
                                      args=(self._toWorkerQ, self._fromWorkerQ, absCheckPath, absVocabPath))

        # Start the worker and wait until it enters prediction cycle
        self._workerProc.start()
        print(self._fromWorkerQ.get())

    def predictNumpyFrame(self, frame):  # Blocking call
        # THERE MUST BE A WAY TO DO THIS WITHOUT WRITING TO DISK!?
        cv2.imwrite("CurrentFrame.jpg", frame)
        # print("Predicting on new image")
        #time.sleep(1)
        f = open("CurrentFrame.jpg", 'rb')
        img = f.read()
        f.close()
        return self.predictOnImage(img)

    def predictOnImage(self, img):  # Blocking call
        self._toWorkerQ.put(img)
        return self._fromWorkerQ.get()
