import sys
import Network.Consumer.Node as Consumer
import Demo.CameraStream.Utils as Utils
import Demo.Img2Fashion.utils as utils
import cv2
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from skimage.measure import find_contours
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import colorsys

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


# Code borrowed from: https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        fig, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    return fig, ax

def main():

    consumerNode = Consumer.Node('194.218.229.164', 1235)
    #consumerNode = Consumer.Node('localhost', 1235)
    out = Utils.initOpenCV('X264', 'output.avi')
    cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
    fullscreen = False

    dataset_class_names = ['BG', 'bag', 'belt', 'boots', 'footwear',
            'coat/jacket/suit/blazers/cardigan/sweater/Jumpsuits/Rompers/vest',
            'dress/t-shir dress', 'sunglasses', 'pants/jeans/leggings',
            'top/blouse/t-shirt/shirt', 'shorts', 'skirt', 'headwear',
            'scarf/tie']

    while (True):
        key=cv2.waitKey(1) & 0xff
        if (key == ord('q')):
            break
        if (key == ord('f')):
            fullscreen = not fullscreen
            print("fullscreen: ", fullscreen)

        r = pickle.loads(consumerNode.getUpdate())
        print(r['masks'].shape)

        (_,_,N) = r['masks'].shape
        if N > 0:
            blank_image = np.zeros(r['masks'][:,:,0].shape + (3,))
            fig, ax = display_instances(blank_image, r['rois'], r['masks'],
                    r['class_ids'], dataset_class_names, r['scores'])

            # redraw the canvas
            fig.canvas.draw()

            # convert canvas to image
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # img is rgb, convert to opencv's default bgr
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

            if fullscreen:
                cv2.setWindowProperty("frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
                cv2.imshow('frame', img)
            else:
                cv2.imshow("frame", img)
        else:
            print("no clothes detected ...")


if (__name__ == '__main__'):
    main()
