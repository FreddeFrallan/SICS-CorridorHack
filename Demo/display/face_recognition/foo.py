import cv2
import os

import get_faces
images = get_faces.get_images('sics')

from face_recog import *

w = start_worker(worker)
to_worker, from_worker, poll_worker = w

for k in images:
    fn = '/tmp/foo.3928749328'
    with open(fn, 'wb') as f: f.write(images[k])
    img = cv2.imread(fn)
    os.remove(fn)
    
    print(img.shape)
    to_worker(('img', img))
    print(k)
    while poll_worker() == False:
        print('.',end='')
        import time
        time.sleep(0.01)
    result = from_worker()

