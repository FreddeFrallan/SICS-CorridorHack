import sys
import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np

import Demo.Img2Fashion.keras_applications as KA
from Demo.Img2Fashion.model import FPN

KA.set_keras_submodules(
    backend=keras.backend,
    layers=keras.layers,
    models=keras.models,
    engine=keras.engine,
    utils=keras.utils,
)

CHECKPOINT_PATH = "data/densenet169-fpn.hdf5"
MODEL_CONFIGURATION = {
    'backbone_name'   : 'densenet169',
    'encoder_weights' : 'imagenet',
    'dropout'         : 'None',
    'drop_block_size' : 1,
    'freeze_encoder'  : False,
    'activation'      : 'softmax',
    'nb_classes'      : 23,
    'image_width'     : 416,
    'image_height'    : 608,
    'nb_channels'     : 3,
    'use_dense_aspp'  : False,
    'return_class_targets'       : False,
    'self_attention_fpn_layers'  : [],
    'pyramid_block_filters'      : 256,
    'segmentation_block_filters' : 128,
}

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

def pad_and_crop(in_tensor, out_cols, out_rows):
    (in_cols, in_rows, in_chs) = in_tensor.shape

    diff_cols = out_cols-in_cols
    diff_rows = out_rows-in_rows

    # center crop
    if diff_cols < 0:
        in_tensor = in_tensor[diff_cols//2:-diff_cols//2,:,:]
    if diff_rows < 0:
        in_tensor = in_tensor[:,diff_rows//2:-diff_rows//2,:]

    (in_cols, in_rows, in_chs) = in_tensor.shape
    diff_cols = out_cols-in_cols
    diff_rows = out_rows-in_rows

    out_tensor = zero_pad(in_tensor, diff_cols//2, diff_rows//2)
    return out_tensor

def zero_pad(tensor, cols_pad, rows_pad):
    (cols, rows, chs) = tensor.shape
    tensor_pad = np.zeros((cols+cols_pad*2, rows+rows_pad*2, chs)).astype(np.uint8)
    tensor_pad[cols_pad:-cols_pad, rows_pad:-rows_pad, :] = tensor

    return tensor_pad

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

def main():
    #model = fpn_keras.load(MODEL_CONFIGURATION)
    model = FPN(
        backbone_name='densenet169',
        input_shape=(608,416,3),
        classes=23
    )
    model.load_weights(CHECKPOINT_PATH)
    frame = cv2.imread(sys.argv[1])
    # BGR -> RGB
    frame = frame[:,:,::-1]

    frame = crop_to_aspect_ratio_and_resize(frame, 6/4, 608, 416)
    input_image = np.copy(frame)
    frame = np.expand_dims(frame, axis=0)
    frame = KA.densenet.preprocess_input(frame)

    prediction = model.predict(frame)
    prediction = np.argmax(prediction, axis=3)[0].astype(np.uint8)
    prediction = seg2rgb(prediction)

    fig, axarr = plt.subplots(1, 2)
    axarr[0].imshow(input_image)
    axarr[1].imshow(prediction)
    plt.show()

if (__name__ == '__main__'):
    main()


