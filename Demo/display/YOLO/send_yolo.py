from use_yolo import start_yolo_thread

import sys
_, displayhost, displayport, frame_nr, videosource = sys.argv
displayport = int(displayport)
frame_nr = int(frame_nr)

yolo = start_yolo_thread()
from yolo_tf import drawOnImage

import vstream
vid = vstream.YoutubeBuffer(videosource)
vid.start()

### prepare to send the image to the display
import listener_t as listener
import pickle
s = listener.Sender(host=displayhost, port=displayport)
def send_img(img):
    s.send(frame_nr, pickle.dumps(img))

import time
tnext = time.time()
fps = 30

result = None
objects = []

while True:
    time.sleep(max(0, tnext - time.time()))
    tnext += 1/fps

    if len(vid.buf) == 0:
        continue

    image = vid.buf[0]

    dt = tnext - time.time()
    if dt < -0.1:
        print(f'behind {dt:.3f}')
        continue

    if result == None:
        result = yolo(image)
    if result.done():
        objects = result.result()
        result = None
        print(' '.join(x[0][0] for x in objects))

    drawOnImage(image, objects)
    # send result to display
    send_img(image)

