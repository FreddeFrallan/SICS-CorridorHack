import multiprocessing
import sys
import listener_t as listener
import vstream
import time
import pickle
import cv2
import os

path = os.path.dirname(os.path.realpath(__file__))
CHECKPOINT_PATH = f"{path}/model/checkpointNew"
VOCABULARY_PATH = f"{path}/model/word_counts.txt"

def mkPredictor():
    import tensorflow as tf
    from Demo.Img2Txt.im2txt.im2txt.inference_utils import caption_generator
    from Demo.Img2Txt.im2txt.im2txt.inference_utils import vocabulary
    from Demo.Img2Txt.im2txt.im2txt import inference_wrapper
    from Demo.Img2Txt.im2txt.im2txt import configuration

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
            # Ignore begin and end words.
            sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
            yield " ".join(sentence)
    return predict


def worker(queue_img, queue_txt):
    queue_txt.put('Thinking...')

    predict = mkPredictor()

    while True:
        img = queue_img.get()
        print(f'got img {img.shape}')
        topSentence = next(predict(img))
        text = topSentence.replace('<S>', '').replace(' .', '')
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
