import Network.Producer.Node as Producer
import Network.Consumer.Node as Consumer
import Demo.Img2Txt.ModelWorker as Img2Txt
import pickle

CHECKPOINT_PATH = "Data\\checkpointNew"
VOCABULARY_PATH = "Data\\word_counts.txt"


def main():
    consNode = Consumer.Node('localhost', 1234)
    prodNode = Producer.Node(1235)
    worker = Img2Txt.Img2TextWorker(CHECKPOINT_PATH, VOCABULARY_PATH)

    while (True):
        # Assuming that we're listening to a video feed that sends a pickled cv2 numpy frame.
        # Contained within a tuple: pickle.dumps( (frame, ) )
        pickledFrame = consNode.getUpdate()
        frame = pickle.loads(pickledFrame)[0]

        predictedSentences = worker.predictNumpyFrame(frame)
        topSentence = predictedSentences[0]

        print(topSentence)
        prodNode.updateContent(pickle.dumps(topSentence))


if (__name__ == '__main__'):
    main()
