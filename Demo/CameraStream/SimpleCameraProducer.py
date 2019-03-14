'''
Simple camera ProducerNode that streams its camera feed to all of its subscribers.
For now there's no compression on this camera feed which may resulting in large file sizes for the individual frames.
As such here we only stream every third frame, of course a real Node would perform some pre-processing before streaming.

The protocol here is simply to pickle the collected frame (numpy array), and stream it to the clients.
Forcing the clients to reverse this process to uncover the intended frame.

Of course in a real situation Try-Catch expressions would be added to account for weird networking crashes.
'''
import Network.Producer.Node as Producer
import cv2, pickle
import Demo.CameraStream.Utils as Utils


# Update the current content on the ProducerNode, using Pickle to convert the frame int bytes.
def updateContent(prodNode, newFrame):
    data = pickle.dumps((newFrame,))
    prodNode.updateContent(data)


def main():
    # Create our producer node on the network, allowing multiple clients to join
    prodNode = Producer.Node(1234)

    # CameraStream initalization
    cap = cv2.VideoCapture(0)
    out = Utils.initOpenCV('X264', 'output.avi')

    frameCounter = 0
    while (True):
        ret, frame = cap.read()
        out.write(frame)

        # Due to the file size of the uncompressed frame, we only send every third frame.
        if (frameCounter % 3 == 0):
            updateContent(prodNode, frame)
        frameCounter += 1

        cv2.imshow('frame', frame)
        if (cv2.waitKey(1) & 0xff == ord('q')):
            break

    # CameraStream cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if (__name__ == '__main__'):
    main()
