from im2txt.inference_utils import vocabulary, caption_generator
import im2txt.inference_wrapper
import im2txt.configuration
import tensorflow as tf
import cv2

def init_im2txt():
    vocab = vocabulary.Vocabulary('model/word_counts.txt')
    g = tf.Graph()
    with g.as_default():
        model = im2txt.inference_wrapper.InferenceWrapper()
        config = im2txt.configuration.ModelConfig()
        restore_fn = model.build_graph_from_config(config, 'model/checkpointNew')
    g.finalize()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25,
                                allow_growth=True)
    sess = tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options))
    sess = tf.Session(graph=g)
    restore_fn(sess)  # Load the model from checkpoint.

    generator = caption_generator.CaptionGenerator(model, vocab)

    def img2txt(image):
        image = cv2.imencode('.jpg', image)[1].tobytes()
        for caption in generator.beam_search(sess, image):
            # Ignore begin and end words.
            sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
            yield " ".join(sentence).replace('<S>', '')

    return img2txt

if __name__ == '__main__':
    import sys
    image = cv2.imread(sys.argv[1])
    im2txt = init_im2txt()
    for i in range(100):
        print(next(im2txt(image)))
