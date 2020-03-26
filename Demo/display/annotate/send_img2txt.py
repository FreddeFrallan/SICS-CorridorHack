import imgToText
import threading
import time
import PIL.ImageFont
import PIL.ImageDraw
import PIL.Image
import pickle
import numpy as np
import argparse

import vstream
import listener_t as listener

args = argparse.ArgumentParser()
args.add_argument('--url')
args.add_argument('--frame_nr', type=int)
args.add_argument('--remote_display', default='localhost:5001')
args.add_argument('--font', default='Consolas Bold Italic.ttf')
args.add_argument('--fontsize', default=40, type=int)
args = args.parse_args()

#url = 'https://www.youtube.com/watch?v=34detVy-Hiw' # math
#url = "https://www.youtube.com/watch?v=MjyDHXOUdGc"

host, port = args.remote_display.split(':')
port = int(port)
args.frame_nr = int(args.frame_nr)
sender = listener.Sender(host, port)
def send_to_display(img):
    sender.send(args.frame_nr, pickle.dumps(img))

fps = 10

if 'youtube' in args.url:
    video = vstream.YoutubeBuffer(args.url)
else:
    video = vstream.VideoBuffer(args.url)

video.start()

im2txt = imgToText.init_im2txt()




text = 'Thinking...'
font = PIL.ImageFont.truetype(args.font, size=args.fontsize)



def update_text():
    while True:
        global text
        text = next(im2txt(video.buf[0]))


t = threading.Thread(target=update_text)
t.start()


def drawText(img, text, x, y, font):
    im = PIL.Image.fromarray(img)
    draw = PIL.ImageDraw.ImageDraw(im)
    draw.text((x, y), text, font=font, fill=(255,255,255,255))
    return np.array(im)


def draw_text_and_send():
    global text, font
    if len(video.buf) == 0: return
    img = video.buf[0]
    textwidth = 30
    pos = ([textwidth]+[i for i,x in enumerate(text) if i < textwidth and x == ' '])[-1]
    text1 = text[:pos]
    text2 = text[pos:]
    img = drawText(img, text1, 20, 200, font)
    img = drawText(img, text2, 20, 240, font)
    #img = drawText(img, text, 20, img.shape[0]-100, font)
    print(f'{text}')
    send_to_display(img)


def every(t, f):
    tnext = time.time()
    while True:
        time.sleep(max(0, tnext-time.time()))
        tnext += t
        yield f()


for x in every(1/fps, draw_text_and_send):
    pass

