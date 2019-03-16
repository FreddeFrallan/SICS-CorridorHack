import Network.Consumer.Node as Consumer
import Demo.CameraStream.Utils as Utils
import cv2, pickle
import numpy as np


# TODO: Some fancy pre-processing of the text..
def paintPicture(sentence, font, out):
    img = np.zeros((1000, 1580, 3), np.uint8)
    cv2.putText(img, sentence, (10, 500), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    out.write(img)
    return img


def main():
    consNode = Consumer.Node('localhost', 1235)
    font = cv2.FONT_HERSHEY_SIMPLEX
    out = Utils.initOpenCV('X264', 'output.avi')

    while (True):
        sentence = pickle.loads(consNode.getUpdate())
        print(sentence)

        cv2.imshow('frame', paintPicture(sentence, font, out))
        if (cv2.waitKey(1) & 0xff == ord('q')):
            break


if (__name__ == '__main__'):
    main()
