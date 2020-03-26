import pickle
import time
import urllib
import listener_t as listener
import cv2
import socket


host = 'localhost'
port = 5000
name = '0'

url = "C:\\Users\\FreddeFrallan\\Pictures\\Typo.png"
url2 = "C:\\Users\\FreddeFrallan\\Pictures\\Present.png"



with listener.Sender(host=host, port=int(port)) as s:
    img = cv2.imread(url)
    img2 = cv2.imread(url2)
    imgs = [img, img2]

    counter = 0
    while True:
        s.send(name, pickle.dumps(imgs[counter]))
        counter = (counter + 1) % 2
        #time.sleep(0.1)