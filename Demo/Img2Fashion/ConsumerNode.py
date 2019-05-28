import sys
import Network.Consumer.Node as Consumer
import Demo.CameraStream.Utils as Utils
import Demo.Img2Fashion.utils as utils
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import numpy as np
import constants

def invert_image_preprocessing(x):
    x_ = np.zeros(x.shape)
    x_[:,:,:] = x
    x_[...,0] *= 0.229
    x_[...,1] *= 0.224
    x_[...,2] *= 0.225

    x_[...,0] += 0.485
    x_[...,1] += 0.456
    x_[...,2] += 0.406

    x_ *= 255

    return x_.astype(np.uint8)


def unpack_data(data):
    print(data.shape)
    seg = data[:,:,0]
    rgb = data[:,:,1:]
    return utils.seg2rgb(seg), invert_image_preprocessing(rgb)

def plot_legend(fig):
    color_code = constants.get_color_code('modanet')
    class_names = constants.get_class_names('modanet')

    colors_and_labels = zip(color_code, class_names)
    patches = [mpatches.Patch(color=np.array(cl[0])/255, label=cl[1]) for cl in colors_and_labels]
    # put those patched as legend-handles into the legend
    bb = (fig.subplotpars.left, fig.subplotpars.top+0.02, 
                  fig.subplotpars.right-fig.subplotpars.left,.1)
    plt.legend(handles=patches, bbox_to_anchor=bb, mode="expand", 
            loc="upper left", ncol=9, borderaxespad=0., bbox_transform=fig.transFigure, facecolor='none', edgecolor='none')


def main():
    consumerNode = Consumer.Node('localhost', 1235)
    #consumerNode = Consumer.Node('194.218.229.164', 1235)
    out = Utils.initOpenCV('X264', 'output.avi')
    #cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
    #fullscreen = False

    plt.ion()
    fig = plt.figure()
    plt.show()
    data = pickle.loads(consumerNode.getUpdate())
    seg, rgb = unpack_data(data)
    plot_legend(fig)
    plt.imshow(rgb)
    plt.pause(0.001)
    plt.axis('off')

    while (True):
        key=cv2.waitKey(1) & 0xff
        if (key == ord('q')):
            break
        if (key == ord('f')):
            fullscreen = not fullscreen
            print("fullscreen: ", fullscreen)

        data = pickle.loads(consumerNode.getUpdate())
        seg, rgb = unpack_data(data)
        plot_legend(fig)
        plt.imshow(rgb)
        plt.imshow(seg, alpha=0.5)
        plt.pause(0.001)

        #if fullscreen:
        #    cv2.setWindowProperty("frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        #    cv2.imshow('frame', utils.seg2rgb(segmentation))
        #else:
        #    cv2.imshow("frame", utils.seg2rgb(segmentation))

if (__name__ == '__main__'):
    main()
