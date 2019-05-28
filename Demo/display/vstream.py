import cv2
import pafy
import time
import numpy as np
import subprocess as sp

FFMPEG_BIN = "ffmpeg" # on Linux ans Mac OS

import threading

class Buf0():
    def __init__(self, url):
        try: url = int(url)
        except: pass
        self.c = cv2.VideoCapture(url)
        assert self.c.isOpened()
        
        self.fps = self.c.get(cv2.CAP_PROP_FPS)
        self.buf = []
        self.running = True
        

    def read(self):
        ok, img = self.c.read()
        assert ok
        return img
        
    def fill_buffer(self):
        while self.running:
            if len(self.buf) > 400:
                time.sleep(1/10)
                
            image = self.read()
            #image = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
            self.buf.append(image)
            
    def empty_buffer(self):
        t0 = time.time()
        next_frame = t0
        while self.running:
            time.sleep(max(0, next_frame - time.time()))
            if len(self.buf) > 1:
                del self.buf[0]
            next_frame += 1/self.fps

    def start(self):
        self.p1 = threading.Thread(target=self.fill_buffer, daemon=True) 
        self.p2 = threading.Thread(target=self.empty_buffer, daemon=True) 
        self.p1.start()
        self.p2.start()
        
    def stop(self):
        self.running = False
        self.buf = []



class Buf(Buf0):
    def __init__(self, url):
        best = pafy.new(url).getbest()
        u = best.url
        self.res = best.dimensions + (3,)
        self.buf = []
        self.running = True
        self.fps = 30
        
        command = [ FFMPEG_BIN,
            '-nostats', '-hide_banner', '-loglevel', 'panic',
            '-i', u,
            '-f', 'image2pipe',
#            '-pix_fmt', 'rgb24',
            '-pix_fmt', 'bgr24',
            '-vcodec', 'rawvideo', '-']
        self.pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**6, stderr=sp.DEVNULL)
        

    def read(self):
        w, h, d = self.res
        raw_image = self.pipe.stdout.read(w*h*d)
        self.pipe.stdout.flush()
        image =  np.frombuffer(raw_image, dtype='uint8')
        image = image.reshape((h, w, d))
        return image
                
    def stop(self):
        super(Buf, self).stop()
        self.pipe.kill()

if __name__ == '__main__':
    import sys
    url = 'https://www.youtube.com/watch?v=1EiC9bvVGnk'
    url = 'https://www.youtube.com/watch?v=Nxs53pkE2TY'
    try: url = sys.argv[1]
    except: pass
    
    b = Buf(url)
    b.start()

    t0 = time.time()
    next_frame = t0
    fps = 30

    while True:
        print(len(b.buf))
        if len(b.buf) > 0:
            image = b.buf[0]
            cv2.imshow('',image)
            if cv2.waitKey(10) == ord('q'):
                break
        time.sleep(max(0, next_frame - time.time()))
        next_frame += 1/fps


    b.stop()
    
