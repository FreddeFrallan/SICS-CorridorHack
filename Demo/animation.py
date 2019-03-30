import numpy as np
import cv2
import pafy
import time
import threading


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
        (x0, y0), (x1, y1) = ul(t - 5 * i).astype(int), lr(t - 5 * i).astype(int)
        a[y0:y1, x0:x1] = cv2.resize(img.astype(float), (x1-x0,y1-y0))

    return a/255


class StreamLoader():
    def __init__(self, url):
        try:
            url1 = pafy.new(url).getbest().url
            url = url1
        except:
            pass
        
        self.c = cv2.VideoCapture(url)
        self.fps = self.c.get(cv2.CAP_PROP_FPS)
        self.img = self.c.read()[1]
        self.next_frame = time.time()


    def fetchloop(self):
        nr = 0
        while True:
            nr += 1
            time.sleep(max(0, self.next_frame - time.time()))
            self.next_frame += 1 / self.fps
            ok, img = self.c.read()
            if ok:
                self.img = img

    def read(self):
        return self.img


urls = ['https://www.youtube.com/watch?v=X0vK_57vQ7s',
        'https://www.youtube.com/watch?v=1EiC9bvVGnk',
        'https://www.youtube.com/watch?v=rWX6BuHUSdM',
        'https://www.youtube.com/watch?v=Nxs53pkE2TY',
        0,
]

streams = [StreamLoader(url) for url in urls]
processes = [threading.Thread(target=s.fetchloop) for s in streams]
for p in processes: p.start()

t0 = time.time()
next_frame = t0 
done = False

while not done:
    fps = 20
    imgs = [s.read() for s in streams]
    imgs = (imgs + imgs)[:8]
    t = time.time() - t0
    img = merge_images(t, imgs)
    cv2.imshow('', img)
    next_frame += 1/fps

    x = cv2.waitKey(max(1, int(1000*(next_frame - time.time()))))
    if x == ord('q'):
        done = True
        cv2.destroyAllWindows()
        break
