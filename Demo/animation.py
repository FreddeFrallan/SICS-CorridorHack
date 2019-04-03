import numpy as np
import cv2
import pafy
import time
import threading
import vstream


def linear(dx, lt): return dx * lt # 0 <= lt <= 1
def quadr(dx, lt): return dx * lt**2 # 0 <= lt <= 1
def iquadr(dx, lt): return dx * (1-(1-lt)**2) # 0 <= lt <= 1
def wait(dx, lt): return 0

def comb(f, dx, dt):
    def ff(cont):
        def fff(t):
            return f(dx, 1-(dt-t)/dt) if t < dt else dx+cont(t-dt)
        return fff
    return ff


def combine(l):
    ans = lambda t: 0
    for f in l[::-1]:
        ans = comb(*f)(ans)
    return ans

def repeat(f, interval):
    return lambda t: f(t%interval)


upper_left = combine([(iquadr, np.array([0.75, 0]), 1), (wait, 0, 4),
                      (iquadr, np.array([0, 0.25]), 1), (wait, 0, 4),
                      (iquadr, np.array([0, 0.25]), 1), (wait, 0, 4),
                      (iquadr, np.array([0, 0.25]), 1), (wait, 0, 4),
                      (iquadr, np.array([-0.25,0]), 1), (wait, 0, 4),
                      (iquadr, np.array([-0.25,0]), 1), (wait, 0, 4),
                      (iquadr, np.array([-0.25,0]), 1), (wait, 0, 4),
                      (iquadr, np.array([0, -0.75]), 1), (wait, 0, 4)])

lower_right = combine([(iquadr, np.array([0.25, -0.5]), 1), (wait, 0, 4),
                      (iquadr, np.array([0, 0.25]), 1), (wait, 0, 4),
                      (iquadr, np.array([0, 0.25]), 1), (wait, 0, 4),
                      (iquadr, np.array([0, 0.25]), 1), (wait, 0, 4),
                      (iquadr, np.array([-0.25,0]), 1), (wait, 0, 4),
                      (iquadr, np.array([-0.25,0]), 1), (wait, 0, 4),
                      (iquadr, np.array([-0.25,0]), 1), (wait, 0, 4),
                      (iquadr, np.array([0.5, -0.25]), 1), (wait, 0, 4)])

w, h = 1080, 720
ul0 = [0, 0]
lr0 = [w * 0.75, h * 0.75]
def ul(t): return repeat(upper_left, 40)(t) * [w, h] + ul0
def lr(t): return repeat(lower_right, 40)(t) * [w, h] + lr0


def merge_images(t, imgs):
    a = np.zeros((h, w,3))

    for i, img in enumerate(imgs):
        try:
            (x0, y0), (x1, y1) = ul(t - 5 * i).astype(int), lr(t - 5 * i).astype(int)
            a[y0:y1, x0:x1] = cv2.resize(img.astype(float), (x1-x0,y1-y0))
        except:
            print(f'bad image {i}, {img.shape}')

    return a/255


class StreamLoader():
    def __init__(self, url):
        self.url = url
        try:
            url1 = pafy.new(url).getbest().url
            url = url1
        except:
            pass
        
        self.c = cv2.VideoCapture(url)
        self.fps = self.c.get(cv2.CAP_PROP_FPS)
        self.img = self.c.read()[1]
        self.buf = [np.zeros((10,10,3))]
        print(f'my fps is {self.fps}')
        if self.fps == 0:
            self.fps == 15

    def fetchloop(self):
        t0 = next_frame = time.time()
        
        while True:
            time.sleep(max(0, next_frame - time.time()))
            next_frame += 1/self.fps
            ok, img = self.c.read()
            if ok:
                self.buf.append(img)

    def updateloop(self):
        self.next_frame = time.time()
        while True:
            time.sleep(max(0, self.next_frame - time.time()))
            if len(self.buf) > 1:
                del self.buf[0]
            self.next_frame += 1 / self.fps

    def read(self):
        return self.buf[0]


urls = ['https://www.youtube.com/watch?v=X0vK_57vQ7s',
        'https://www.youtube.com/watch?v=1EiC9bvVGnk',
        'https://www.youtube.com/watch?v=rWX6BuHUSdM',
        'https://www.youtube.com/watch?v=Nxs53pkE2TY',
        'https://www.youtube.com/watch?v=DBd560d1DIM',
        #'https://www.youtube.com/watch?v=sX1Y2JMK6g8',
        'https://www.youtube.com/watch?v=JqUREqYduHw',
        0]

cam_stream = 'rtsp://192.168.0.142:7447/5ca3cbe70c0af3aa8fbeab20_2'
cs = StreamLoader(cam_stream)
p1 = threading.Thread(target=cs.fetchloop, daemon=True)
p2 = threading.Thread(target=cs.updateloop, daemon=True)
p1.start()
p2.start()


bs = [vstream.Buf(u) for u in urls if 'youtube' in str(u)]
for b in bs: b.start()


t0 = time.time()
next_frame = t0 
done = False

cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

while not done:
    fps = 30
    imgs = [b.buf[0] for b in bs if len(b.buf)>0]
    imgs.append(cs.read())
    imgs += [np.ones((10,10,3))*30*(i+1) for i in range(8)]
    imgs = imgs[:8]
    t = time.time() - t0
    img = merge_images(t/3, imgs)
    cv2.imshow('window', img)
    next_frame += 1/fps
    for b in bs+[cs]: print(f'{len(b.buf):-4d}, ', end='')
    print()
    x = cv2.waitKey(max(1, int(1000*(next_frame - time.time()))))
    if x == ord('q'):
        done = True
        cv2.destroyAllWindows()
        break

for b in bs: b.stop()
