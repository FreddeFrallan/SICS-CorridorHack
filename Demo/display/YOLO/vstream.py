import cv2
import pafy
import time
import numpy as np
import subprocess as sp
import os

ydl_opts = {'cookiefile':os.environ.get('COOKIES')}

FFMPEG_BIN = "ffmpeg"  # on Linux and Mac OS

import threading


class VideoBuffer():
    def __init__(self, url):
        self.url = url
        self.buf = []
        self.init_input_feed()

    def read(self):
        ok, img = self.c.read()
        assert ok
        return img

    def fill_buffer(self):
        while self.running:
            if len(self.buf) > 400:
                time.sleep(1 / 10)

            try:
                image = self.read()
                self.buf.append(image)
            except:
                self.init_input_feed()
                
    def empty_buffer(self):
        t0 = time.time()
        next_frame = t0
        while self.running:
            time.sleep(max(0, next_frame - time.time()))
            if len(self.buf) > 1:
                del self.buf[0]
                
            next_frame += 1 / self.fps

    def start(self):
        self.p1 = threading.Thread(target=self.fill_buffer, daemon=True)
        self.p2 = threading.Thread(target=self.empty_buffer, daemon=True)
        self.p1.start()
        self.p2.start()

    def stop(self):
        self.running = False

    def init_input_feed(self):
        url = self.url
        try:    url = int(url) # maybe use the local camera
        except: pass

        self.c = cv2.VideoCapture(url)
        assert self.c.isOpened()

        self.fps = self.c.get(cv2.CAP_PROP_FPS)
        self.running = True


class YoutubeBuffer(VideoBuffer):

    def read(self):
        w, h, d = self.res
        raw_image = self.pipe.stdout.read(w * h * d)
        self.pipe.stdout.flush()
        image = np.frombuffer(raw_image, dtype='uint8')
        image = image.reshape((h, w, d))
        return image

    def stop(self):
        super(YoutubeBuffer, self).stop()
        self.pipe.kill()

    def init_input_feed(self):
        try:
            self.pipe.terminate()
            import time; time.sleep(1)
            self.pipe.kill()
        except:
            pass
        
        best = pafy.new(self.url, ydl_opts=ydl_opts).getbest()
        u = best.url
        self.res = best.dimensions + (3,)
        self.fps = 30

        command = [FFMPEG_BIN,
                   '-nostats', '-hide_banner', '-loglevel', 'panic',
                   '-i', u,
                   '-f', 'image2pipe',
                   #            '-pix_fmt', 'rgb24',
                   '-pix_fmt', 'bgr24',
                   '-vcodec', 'rawvideo', '-']
        try:
            self.pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.DEVNULL)            
            self.running = True
        except:
            print(f'Failed to restart input stream, buf={len(self.buf)}')
            if len(self.buf) == 1:
                print(f'stopping...')
                self.stop()

if __name__ == '__main__':
    import sys

    url = 'https://www.youtube.com/watch?v=1EiC9bvVGnk'
    url = 'https://www.youtube.com/watch?v=Nxs53pkE2TY'
    try:
        url = sys.argv[1]
    except:
        pass

    b = YoutubeBuffer(url)
    b.start()

    t0 = time.time()
    next_frame = t0
    fps = 30

    while b.running or len(b.buf)>1:
        if len(b.buf) > 0:
            image = b.buf[0]
            cv2.imshow('', image)
            if cv2.waitKey(10) == ord('q'):
                break
        time.sleep(max(0, next_frame - time.time()))
        next_frame += 1 / fps

    b.stop()
