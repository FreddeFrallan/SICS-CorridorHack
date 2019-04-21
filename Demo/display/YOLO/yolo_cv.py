import cv2
import numpy as np



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


