
def start_yolo_thread():
    import threading
    import queue
    import concurrent

    import yolo_tf
    
    YOLO = yolo_tf.init_yolo()
    
    q = queue.Queue()
    def yolo_worker():
        while True:
            ans, img = q.get()
            ans.set_result(YOLO(img))
    
    p = threading.Thread(target=yolo_worker)
    p.start()

    def send_job(image):
        f = concurrent.futures.Future()
        q.put((f, image))
        return f

    return send_job

if __name__ == '__main__':
    yolo = start_yolo_thread()
    from yolo_tf import drawOnImage
    
    import vstream
    videosource = 'https://www.youtube.com/watch?v=rQSwh3bgs5k'
    vid = vstream.YoutubeBuffer(videosource)
    vid.start()

    
    ### prepare to send the image to frame 5 on the display
    import listener_t as listener
    import pickle
    s = listener.Sender(host='localhost', port=5001)
    def send_img(img):
        s.send(5, pickle.dumps(img))

    import time
    tnext = time.time()
    fps = 30
    
    result = None
    objects = []

    while True:
        time.sleep(max(0, tnext - time.time()))
        tnext += 1/fps
        
        if len(vid.buf) == 0:
            continue
        
        image = vid.buf[0]
        
        dt = tnext - time.time()
        if dt < -0.1:
            print(f'behind {dt:.3f}')
            continue

        if result == None:
            result = yolo(image)
        if result.done():
            objects = result.result()
            result = None
            print(' '.join(x[0][0] for x in objects))

        drawOnImage(image, objects)
        # send result to display
        send_img(image)

