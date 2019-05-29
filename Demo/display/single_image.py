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

to_display = make_sender(frame_nr=frame_nr, host=host, port=port)


import cv2
im = cv2.imread('/home/fredrik/Pictures/DemoPoster.png')

to_display(im)
