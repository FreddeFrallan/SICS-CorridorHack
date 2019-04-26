import multiprocessing as mp
import time
import concurrent
import queue
import numpy as np
import dlib


def dist(a, b):
    return np.mean((a-b)**2, axis=1)


def read_facedatabase(faces_db):
    faces = np.array([]).reshape((0, 128))
    names = []

    with open(faces_db) as f:
        while True:
            name = f.readline().strip()
            if len(name) == 0:
                break
            data = f.readline()
            data = eval(data)
            face = np.array(data).reshape((1, 128))
            faces = np.concatenate([faces, face])
            names.append(name)

    return faces, names


def load_models():
    predictor_path = 'shape_predictor_5_face_landmarks.dat'
    # http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2

    face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
    # http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)
    return detector, shape_predictor, facerec


def get_faces(img, faces, face_model):
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
        result.append((best_match, coords, face_hash, face_chip, err))
    return result


def worker(p_in, p_out):
    global faces, names

    face_model = load_models()

    while True:
        cmd, data = p_in.recv()
        if cmd == 'img':
            img = data
            result = get_faces(img, faces, face_model)
            p_out.send(result)
        if cmd == 'newface':
            face, name = data
            faces = np.concatenate([faces, face])
            names.append(name)


def drawText(image, text, x, y):
    bottomLeftCornerOfText = (x, y)
    cv2.putText(image, text,
                org=bottomLeftCornerOfText,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                lineType=2)


def flush_input():
    try:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
    except ImportError:
        # for linux/unix
        import sys
        import termios
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)


def add_faces_from_dir(dirname, faces, names, face_model):
    import glob
    import os
    import cv2

    detector, shape_predictor, facerec = face_model

    for f in glob.glob(f'{dirname}/*'):
        name = os.path.basename(f)
        name = os.path.splitext(name)[0]
        img = cv2.imread(f)
        d = detector(img, 1)
        if len(d) != 1:
            print(f'error finding faces in {f}, {len(d)} found')
            continue
        d = d[0]

        shape = shape_predictor(img, d)
        face_chip = dlib.get_face_chip(img, shape)
        face_hash = facerec.compute_face_descriptor(face_chip)
        face_hash = np.array(face_hash).reshape((1, 128))

        if dist(faces, face_hash).min() > 0:
            faces = np.concatenate([faces, face_hash])
            names.append(name)
            with open(faces_db, 'a') as f:
                f.write(name+'\n')
                f.write(str(face_hash.tolist())+'\n')
            print(f'added {name} to face database')

    return faces, names

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
    try: args.video = int(args.video)
    except: pass

    # read face database and add more faces from specified dir
    faces_db = 'faces.db'
    faces, names = read_facedatabase(faces_db)
    if args.add_faces:
        face_model = load_models()
        faces, names = add_faces_from_dir(args.add_faces,
                                          faces, names, face_model)

    # prepare to send to display
    def make_sender(frame_nr, host='localhost', port=5001):
        import listener_t as listener
        import pickle
        sender = listener.Sender(host, port)
        def to_display(img):
            sender.send(frame_nr, pickle.dumps(img))
        return to_display

    if ':' in args.remote_display:
        host, port = args.remote_display.split(':')
        port = int(port)
        to_display = make_sender(frame_nr=2, host=host, port=port)
        remote_display = True
    else:
        remote_display = False


    send_job, recv_w = mp.Pipe()
    send_w, get_answer = mp.Pipe()
    # worker needs the global variable face_model to be defined here
    w = mp.Process(target=worker, args=(recv_w, send_w), daemon=True)
    w.start()

    import cv2
    c = cv2.VideoCapture(args.video)
    if not c.isOpened():
        print(f'couldnt open {args.video}')
        import sys
        sys.exit(1)
    
    pending = False
    result = []

    import time
    tnext = time.time()
    fps = 30
    print('starting')
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
            send_job.send(('img', img))
        if pending and get_answer.poll():
            result = get_answer.recv()
            pending = False
            n = 'people: '
            for i, (best_match, (x0, y0, x1, y1), face, face_img, err) \
                in enumerate(result):
                n += f'{names[best_match]}, '
            print(n)

        for i, (best_match, (x0, y0, x1, y1), face, face_img, err) \
            in enumerate(result):
            drawText(img, f'{names[best_match]}', x0, y1)
            # insert faces into image
            #hh, ww, _ = face_img.shape
            #img[0:hh, i * ww:(i + 1) * ww] = face_img

        if remote_display:
            to_display(img)
            
        if args.local_display:
            cv2.imshow('', img)
            key = cv2.waitKey(10)
            if key == ord('q'):
                break
            if key == ord('a'):
                for best_match, (x0, y0, x1, y1), face, face_img, err in result:
                    flush_input()
                    cv2.imshow('', face_img)
                    cv2.waitKey(1)
                    newname = input('Who is this? '
                                    '(just press RETURN to skip this face) : ')
                    if newname != '':
                        faces = np.concatenate([faces, face])
                        names.append(newname)
                        with open(faces_db, 'a') as f:
                            f.write(newname+'\n')
                            f.write(str(face.tolist())+'\n')

                        send_job.send(('newface',(face, newname))) # tell the worker
