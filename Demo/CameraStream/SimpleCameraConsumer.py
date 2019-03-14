'''
Simple camera ConsumerNode that displays the camera feed it receives.

In the unpacking of the collected updates we assume that message was sent from SimpleCameraProduer.
So we reverse the pickle!

Of course in a real situation Try-Catch expressions would be added to account for weird networking crashes.
'''

import Network.Consumer.Node as Consumer
import cv2, pickle
import Demo.CameraStream.Utils as Utils


def main():
    consNode = Consumer.Node('localhost', 1234)
    out = Utils.initOpenCV('X264', 'output.avi')

    while (True):
        # Unpack the collected update, assuming that the frame was sent from SimpleCameraProducer
        pickledFrame = consNode.getUpdate()
        frame = pickle.loads(pickledFrame)[0]

        out.write(frame)
        cv2.imshow('frame', frame)

        if (cv2.waitKey(1) & 0xff == ord('q')):
            break

    out.release()
    cv2.destroyAllWindows()


if (__name__ == '__main__'):
    main()
