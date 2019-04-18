import cv2
import time
import numpy as np
import pafy


net = cv2.dnn.readNetFromDarknet(
    cfgFile='yolo-coco/yolov3.cfg',
    darknetModel='yolo-coco/yolov3.weights')

LABELS = open('yolo-coco/coco.names').read().split('\n')

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def yolo(image):
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)

    net.setInput(blob)

    layerOutputs = net.forward(ln)
    
    return layerOutputs


def getBoxesAndLabels(layerOutputs, W, H, confidence_limit=0.5, nm_threshold=0.3):
    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidence_limit:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_limit,
                            nm_threshold)


    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            id = classIDs[i]
            label = LABELS[id]
            confidence = confidences[i]

            yield (label, confidence), (x, y, w, h)



def INIT_COLORS(labels):
    def rgb(): return np.random.randint(0, 255, size=3)
    return {l:rgb() for l in labels}
    
COLORS = INIT_COLORS(LABELS)


def drawOnImage(image, markings):
    for (label, confidence), (x, y, w, h) in markings:
        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in COLORS[label]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = f"{label}: {confidence:.4f}"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)


def YOLO(image):
    H, W = image.shape[:2]
    return list(getBoxesAndLabels(yolo(image), W, H))



if __name__ == '__main__':

    import pylab as pl
    pl.axis('off')
    pl.ion()
    
    import sys
    videosource = 'videos/car_chase_01.mp4'
    videosource = 'https://www.youtube.com/watch?v=rQSwh3bgs5k'

    import vstream
    if 'youtube' in videosource:
        vid = vstream.YoutubeBuffer(videosource)
    else:
        vid = vstream.VideoBuffer(videosource)

    # send the image to frame 5 on the display
    import listener_t as listener
    import pickle
    s = listener.Sender(host='localhost', port=5001)
    def send_img(img): s.send(5, pickle.dumps(img))
    
    vid.start()
        
    tnext = time.time()
    fps = 30

    import multiprocessing as mp
    pool = mp.Pool()
    result = None
    objects = []
    while True:
        
        tnext += 1/fps
                 
        #ok, image = vid.read()
        #if not ok: break
        if len(vid.buf) == 0:
            continue
        
        image = vid.buf[0]
        
        dt = tnext - time.time()
        if dt < -0.1:
            print(f'behind {dt:.3f}')
            continue

        if result == None:
            result = pool.map_async(YOLO, [image])
            
        if result.ready():
            objects = result.get()[0]
            result = None
            
        #image = image[::3, ::3].copy()
        #objects = YOLO(image)
        #image = np.zeros_like(image)
        drawOnImage(image, objects)

        # send result to display
        send_img(image)

        # draw on screen with pylab
	# pl.pause(max(0.001, dt))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # pl.imshow(image)

        # draw on screen with opencv
        #cv2.imshow('', image)
        #cv2.waitKey(min(1, int(1000*dt)))

