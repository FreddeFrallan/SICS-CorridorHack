import cv2
import numpy as np

import models.keras_applications as KA

def prepare_cv2_frame(frame):
    frame = frame[:,:,::-1]
    frame = crop_to_aspect_ratio_and_resize(frame, 6/4, 608, 416)
    frame = np.expand_dims(frame, axis=0)
    frame = KA.densenet.preprocess_input(frame)

    return frame

def crop_to_aspect_ratio_and_resize(in_tensor, out_ratio, out_cols, out_rows):
    (in_rows, in_cols, in_chs) = in_tensor.shape

    in_ratio = in_rows/in_cols

    if in_ratio < out_ratio:
        # crop rows
        cols = in_rows/out_ratio
        diff_cols = int(in_cols-cols)
        out_tensor = in_tensor[:, diff_cols//2:-diff_cols//2,:]
    elif in_ratio > out_ratio:
        # crop columns
        rows = in_cols*out_ratio
        diff_rows = int(in_rows-rows)
        out_tensor = in_tensor[diff_rows//2:-diff_rows//2,:,:]
    else:
        out_tensor = in_tensor

    out_tensor = cv2.resize(out_tensor, (out_rows, out_cols), cv2.INTER_NEAREST)
    return out_tensor

def seg2rgb(seg):
    color_code = [
        [255,255,255],
        [226,196,196],
        [64,32,32],
        [255,0,0],
        [255,70,0],
        [255,139,0],
        [255,209,0],
        [232,255,0],
        [162,255,0],
        [93,255,0],
        [23,255,0],
        [0,255,46],
        [0,255,116],
        [0,255,185],
        [0,255,255],
        [0,185,255],
        [0,116,255],
        [0,46,255],
        [23,0,255],
        [93,0,255],
        [162,0,255],
        [232,0,255],
        [255,0,209],
        [255,0,139],
        [255,0,70]
    ]

    rgb = np.zeros(seg.shape + (3,)).astype(np.uint8)

    for idx_class in range(len(color_code)):
        mask = seg == idx_class
        rgb[mask,:] = color_code[idx_class]

    return rgb


