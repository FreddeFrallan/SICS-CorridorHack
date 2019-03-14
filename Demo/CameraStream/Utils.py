import cv2

def initOpenCV(codec, outputName):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(outputName, fourcc, 60, (640, 400))

