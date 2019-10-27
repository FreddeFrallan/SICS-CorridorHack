import numpy as np
import time
import pickle
import sys
import cv2
import listener_t as listener

# ******************** SETTINGS *******************
SLIDE_TIME = 0.3
WAIT_TIME = 8
SLOT_TIME = SLIDE_TIME + WAIT_TIME

WIDTH, HEIGHT = 1920, 1080
UL_0 = [0, 0]
LR_0 = [WIDTH * 0.75, HEIGHT * 0.75]


# ***************************************************


def linear(dx, dt): return dx * dt  # 0 <= lt <= 1


def quadr(dx, dt): return dx * dt ** 2  # 0 <= lt <= 1


def iquadr(dx, dt): return dx * (1 - (1 - dt) ** 2)  # 0 <= lt <= 1


def wait(dx, dt): return 0


def comb(f, dx, dt):
    def ff(cont):
        def fff(t):
            return f(dx, 1 - (dt - t) / dt) if t < dt else dx + cont(t - dt)

        return fff

    return ff


def combine(l):
    ans = lambda t: 0
    for f in l[::-1]:
        ans = comb(*f)(ans)
    return ans


def repeat(f, interval):
    return lambda t: f(t % interval)


def createAnimationCycle(slideTime, waitTime):
    cycleUL = [(iquadr, np.array([0.75, 0]), slideTime), (wait, 0, waitTime),
               (iquadr, np.array([0, 0.25]), slideTime), (wait, 0, waitTime),
               (iquadr, np.array([0, 0.25]), slideTime), (wait, 0, waitTime),
               (iquadr, np.array([0, 0.25]), slideTime), (wait, 0, waitTime),
               (iquadr, np.array([-0.25, 0]), slideTime), (wait, 0, waitTime),
               (iquadr, np.array([-0.25, 0]), slideTime), (wait, 0, waitTime),
               (iquadr, np.array([-0.25, 0]), slideTime), (wait, 0, waitTime),
               (iquadr, np.array([0, -0.75]), slideTime), (wait, 0, waitTime)]

    cycleLR = [(iquadr, np.array([0.25, -0.5]), slideTime), (wait, 0, waitTime),
               (iquadr, np.array([0, 0.25]), slideTime), (wait, 0, waitTime),
               (iquadr, np.array([0, 0.25]), slideTime), (wait, 0, waitTime),
               (iquadr, np.array([0, 0.25]), slideTime), (wait, 0, waitTime),
               (iquadr, np.array([-0.25, 0]), slideTime), (wait, 0, waitTime),
               (iquadr, np.array([-0.25, 0]), slideTime), (wait, 0, waitTime),
               (iquadr, np.array([-0.25, 0]), slideTime), (wait, 0, waitTime),
               (iquadr, np.array([0.5, -0.25]), slideTime), (wait, 0, waitTime)]

    def calcAnimationRuntime(animationCycle):
        return np.sum([a[-1] for a in animationCycle])

    time1 = calcAnimationRuntime(cycleUL)
    time2 = calcAnimationRuntime(cycleLR)
    assert time1 == time2
    totalCycleTime = time1

    ul = lambda t: repeat(upper_left, totalCycleTime)(t) * [WIDTH, HEIGHT] + UL_0
    lr = lambda t: repeat(lower_right, totalCycleTime)(t) * [WIDTH, HEIGHT] + LR_0

    return combine(cycleUL), combine(cycleLR), ul, lr


upper_left, lower_right, ul, lr = createAnimationCycle(SLIDE_TIME, WAIT_TIME)


def merge_images(t, imgs, qualityControl):
    a = np.zeros((HEIGHT, WIDTH, 3))

    totTime = SLOT_TIME * 8
    for i, img in enumerate(imgs):
        try:
            t1 = t - SLOT_TIME * i
            currentSlot = np.floor((t1 % totTime) / SLOT_TIME)
            if(currentSlot == 7):
                qualityControl[i].setHighQuality()
            else:
                qualityControl[i].setLowQuality()

            (x0, y0), (x1, y1) = ul(t1).astype(int), lr(t1).astype(int)
            a[y0:y1, x0:x1] = cv2.resize(img.astype(float), (x1 - x0, y1 - y0))
        except:
            print(f'bad image {i}, {img.shape}')

    return a / 255


def start_animation(images, qualityControl):
    t0 = time.time()
    next_frame = t0
    done = False

    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while not done:
        fps = 30
        imgs = [np.ones((10, 10, 3)) * 30 * (i + 1) for i in range(8)]
        for i in range(8):
            if str(i) in images:
                imgs[i] = pickle.loads(images[str(i)])

        t = time.time() - t0
        img = merge_images(t / 2, imgs, qualityControl)
        cv2.imshow('window', img)
        next_frame += 1 / fps

        try:
            x = cv2.waitKey(max(1, int(1000 * (next_frame - time.time()))))
        except:
            pass
        if x == ord('q'):
            done = True
            cv2.destroyAllWindows()
            break


class StreamQuality:
    def __init__(self, id):
        self.quality = "Low"
        self.id = id

    def setHighQuality(self):
        self.quality = "High"

    def setLowQuality(self):
        self.quality = "Low"


if __name__ == '__main__':
    try:
        host, port = sys.argv[1:]
    except:
        host = "localhost"  # input("Host: ")
        port = 5001  # int(input("Port: "))
    images = {}
    qualityControl = [StreamQuality(i) for i in range(8)]
    listener.start_listener(host, int(port), images, qualityControl)

    time.sleep(60)
    start_animation(images, qualityControl)
