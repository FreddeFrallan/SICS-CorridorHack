import multiprocessing as mp
import time
import os
import glob
import concurrent
import queue
import numpy as np
import dlib
import newfiles
import cv2

def dist(a, b):
    return np.mean((a-b)**2, axis=1)


def load_models():
    predictor_path = 'shape_predictor_5_face_landmarks.dat'
    # http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2

    face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
    # http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)
    return detector, shape_predictor, facerec


def get_faces(img, faces, names, face_model):
    detector, shape_predictor, facerec = face_model
    result = []
    detected = detector(img, 1)
    for d in detected:
        shape = shape_predictor(img, d)
        face_chip = dlib.get_face_chip(img, shape)
        face_hash = facerec.compute_face_descriptor(face_chip)
        face_hash = np.array(face_hash).reshape((1, 128))
        distances = dist(faces, face_hash)
        best_match = np.argmin(distances)
        err = np.min(distances)
        r = shape.rect
        coords = (r.left(), r.top(), r.right(), r.bottom())
        result.append((names[best_match],
                       coords, face_hash, face_chip, err))
    return result



def start_worker(worker):
    send_job, recv_w = mp.Pipe()
    send_w, get_answer = mp.Pipe()

    w = mp.Process(target=worker, args=(recv_w, send_w),
                   daemon=True)
    w.start()

    return send_job.send, get_answer.recv, get_answer.poll


def worker(p_in, p_out):
    faces = np.zeros((1,128))
    names = ['zero']

    face_model = load_models()

    while True:
        cmd, data = p_in.recv()
        if cmd == 'img':
            img = data
            result = get_faces(img, faces, names, face_model)
            p_out.send(result)
        if cmd == 'newface':
            face, name = data
            faces = np.concatenate([faces, face])
            names.append(name)


def add_faces(d, w):
    to_worker, from_worker, poll_worker = w

    faces, names = [], []
    
    for fn in glob.glob(f'{d}/*'):
        name = os.path.basename(fn)
        name = os.path.splitext(name)[0]
        img = cv2.imread(fn)
        os.remove(fn)
        if img is None:
            continue
        to_worker(('img', img))
        result = from_worker()
        if len(result) == 1:
            _, coords, face_hash, face_chip, err = result[0]

            print(name, err)
            if err != 0:
                face_hash = face_hash.reshape((1, 128))
                to_worker(('newface',(face_hash, name)))
                yield face_hash, name


def drawText(image, text, x, y):
    bottomLeftCornerOfText = (x, y)
    cv2.putText(image, text,
                org=bottomLeftCornerOfText,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                lineType=2)


def drawText(img, text, x, y, font):
    import PIL.ImageDraw
    import PIL.Image
    im = PIL.Image.fromarray(img)
    draw = PIL.ImageDraw.ImageDraw(im)
    # fontname = 'Consolas Bold Italic.ttf'
    # font = PIL.ImageFont.truetype(fontname, size=40)
    draw.text((x, y), text, font=font)
    return np.array(im)


def flush_stdin():
    try:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
    except ImportError:
        # for linux/unix
        import sys
        import termios
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)


def read_two(g):
    for a in g: yield a, next(g)


def tryint(x):
    try: return int(x)
    except: return x

######################################################################



if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    #args.add_argument('--video', default='0')
    cam_stream = 'rtsp://192.168.0.142:7447/5ca3cbe70c0af3aa8fbeab20_2'
    cam_stream = 'rtsp://192.168.0.142:7447/5ca3cbe70c0af3aa8fbeab20_1'

    args.add_argument('--video', default=cam_stream)

    args.add_argument('--add_faces', default='')
    args.add_argument('--remote_display', default='localhost:5001')
    args.add_argument('--local_display', default=False, action='store_true')

    args = args.parse_args()
    args.video = tryint(args.video)
    # cv2 opens a local camera if it is an int

    ####### ASYNCHRONOUS FACE RECOGNITION PROCESS
    w = start_worker(worker)
    to_worker, from_worker, poll_worker = w

    ######## FACES DATABASE
    # read face database and add more faces from specified dir
    faces_db = 'faces.db'
    for fdb in [faces_db, 'acreo.db']:
        with open(fdb) as f:
            for name, face in read_two(f):
                face = np.array(eval(face)).reshape((1, 128))
                to_worker(('newface', (face, name)))

    last_face_check = 0

    
    ####### REMOTE DISPLAY
    # prepare to send to display
    def make_sender(frame_nr, host='localhost', port=5001):
        import listener_t as listener
        import pickle
        sender = listener.Sender(host, port)
        def to_display(img):
            sender.send(frame_nr, pickle.dumps(img))
        return to_display

    args.remote_display = ''
    if ':' in args.remote_display:
        host, port = args.remote_display.split(':')
        port = int(port)
        to_display = make_sender(frame_nr=2, host=host, port=port)
        remote_display = True
    else:
        remote_display = False

    ######## INPUT SOURCE OF IMAGES
    import cv2
    c = cv2.VideoCapture(args.video)
    if not c.isOpened():
        print(f'couldnt open {args.video}')
        import sys
        sys.exit(1)

    ########## FOR DRAWING TEXT ON IMAGE
    import PIL.ImageFont
    fontname = 'Consolas Bold Italic.ttf' 
    font = PIL.ImageFont.truetype(fontname, size=40)


    #################################
    pending = False
    result = []

    tnext = time.time()
    fps = 30
    print(f'starting')
    while True:
        time.sleep(max(0, tnext-time.time()))
        tnext += 1/fps

        if not c.isOpened():
            break

        ok, img = c.read()
        if not ok:
            break

        if not pending:
            pending = True
            to_worker(('img', img))
        if pending and poll_worker():
            result = from_worker()
            pending = False
            names = []
            for i, (name, (x0, y0, x1, y1),
                    face, face_img, err) in enumerate(result):
                names.append(name)

            if len(names) > 0: print('\n' + (', '.join(names)))
            else: print('.',end='', flush=True)

        for i, (name, (x0, y0, x1, y1), face, face_img,
                err) in enumerate(result):
            img = drawText(img, f'{name}', x0, y1, font)
            # insert faces into image
            # hh, ww, _ = face_img.shape
            # img[0:hh, i * ww:(i + 1) * ww] = face_img

        if remote_display:
            to_display(img)

        ######### SEE IF NEW FACE IMAGES HAVE BEEN ADDED
        if args.add_faces:
            ctime = os.stat(args.add_faces).st_ctime
            if ctime > last_face_check:
                if pending:
                    _ = from_worker()
                    pending = False
                last_face_check = ctime
                with open(faces_db, 'a') as f:
                    for face, name in add_faces(d=args.add_faces, w=w):
                        to_worker(('newface', (face, name)))
                        f.write(name+'\n')
                        f.write(str(face.tolist())+'\n')

        if args.local_display:
            cv2.imshow('', img)
            key = cv2.waitKey(10)
            if key == ord('q'):
                break
            if key == ord('a'):
                for best_match, _, (x0, y0, x1, y1), \
                    face, face_img, err in result:
                    flush_stdin()
                    cv2.imshow('', face_img)
                    cv2.waitKey(1)
                    name = input('Who is this? '
                                    '(just press RETURN to skip this face) : ')
                    if name != '':
                        with open(faces_db, 'a') as f:
                            f.write(newname+'\n')
                            f.write(str(face.tolist())+'\n')

                        to_worker(('newface', (face, name)))
