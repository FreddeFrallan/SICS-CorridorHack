import numpy as np
import time
import threading
import pickle
import sys
import cv2
import pafy

import vstream
import listener_t as listener

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
            cv2.resize(img.astype(float), (x1-x0,y1-y0), a[y0:y1, x0:x1])
        except:
            print(f'bad image {i}, {img.shape}')

    return a/255


def start_animation(images):
    
    t0 = time.time()
    next_frame = t0 
    done = False

    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    while not done:
        fps = 30
        imgs = [np.ones((10,10,3))*30*(i+1) for i in range(8)]
        for i in range(8):
            if str(i) in images:
                imgs[i] = pickle.loads(images[str(i)])

        t = time.time() - t0
        img = merge_images(t*2, imgs)
        cv2.imshow('window', img)
        next_frame += 1/fps

        x = cv2.waitKey(max(1, int(1000*(next_frame - time.time()))))
        if x == ord('q'):
            done = True
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    host, port = sys.argv[1:]
    images = {}
    listener.start_listener(host, int(port), images)
    start_animation(images)
    
