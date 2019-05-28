import listener_t as listener
import sys
import vstream
import time
import pickle

host, port, frame, url = sys.argv[1:]

with listener.Sender(host=host, port=int(port)) as s:
    if 'youtube' in url:
        b = vstream.Buf(url)
    else:
        b = vstream.Buf0(url)
        
    b.start()

    t0 = time.time()
    next_frame = t0
    fps = 30

    while True:
        if len(b.buf) > 0:
            s.send(frame, pickle.dumps(b.buf[0]))
        time.sleep(max(0, next_frame - time.time()))
        next_frame += 1/fps

    b.stop()


