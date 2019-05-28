import cv2
import numpy as np
import model
import constants

import keras_applications as KA

def prepare_cv2_frame(frame, backbone_name):
    frame = frame[:,:,::-1]
    frame = crop_to_aspect_ratio_and_resize(frame, 6/4, 608, 416)
    frame = np.expand_dims(frame, axis=0)
    preprocess_input = model.get_preprocessing(backbone_name)
    frame = preprocess_input(frame)

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
    color_code = constants.get_color_code('modanet')
    rgb = np.zeros(seg.shape + (3,)).astype(np.uint8)

    for idx_class in range(len(color_code)):
        mask = seg == idx_class
        rgb[mask,:] = color_code[idx_class]

    return rgb


