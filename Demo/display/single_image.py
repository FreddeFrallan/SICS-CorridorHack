host, port, frame_nr = 'localhost', 5001, 0
####### REMOTE DISPLAY
# prepare to send to display
def make_sender(frame_nr, host='localhost', port=5001):
    import listener_t as listener
    import pickle
    sender = listener.Sender(host, port)
    def to_display(img):
        sender.send(frame_nr, pickle.dumps(img))
    return to_display



frame_nr = 0
filename = '/home/fredrik/Pictures/DemoPoster.png'

import sys
frame_nr = frame_nr if len(sys.argv)<3 else int(sys.argv[2])
filename = filename if len(sys.argv)<2 else sys.argv[1]

import cv2
im = cv2.imread(filename)

to_display = make_sender(frame_nr=frame_nr, host=host, port=port)

to_display(im)
