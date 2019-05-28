import sys
import time
import pickle
import listener_t as listener
import vstream
import img2txt

host, port, name, url = sys.argv[1:]


with listener.Sender(host=host, port=int(port)) as s:
    if 'youtube' in url or 'youtu.be' in url:
        b = vstream.Buf(url)
    else:
        b = vstream.Buf0(url)
        
    b.start()

    t0 = time.time()
    next_frame = t0
    fps = 30
    text = ''
    while True:
        
        if len(b.buf) > 0:
            img = b.buf[0]
            oldtext = text
            text = img2txt.annotate(img, text)
            if oldtext != text: print(text)
            s.send(name, pickle.dumps(img))

        time.sleep(max(0, next_frame - time.time()))
        next_frame += 1/fps

    b.stop()


