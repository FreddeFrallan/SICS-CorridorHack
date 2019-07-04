import torch
import numpy as np
import cv2
import sys
sys.path.insert(0,'posenet-pytorch')
import posenet



####### REMOTE DISPLAY
# prepare to send to display
def make_sender(frame_nr, host='localhost', port=5001):
    import listener_t as listener
    import pickle
    sender = listener.Sender(host, port)
    def to_display(img):
        sender.send(frame_nr, pickle.dumps(img))
    return to_display


to_display = make_sender(frame_nr=3, host='localhost', port=5001)



def valid_resolution(width, height, output_stride=16):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height


def prepare_img(source_img, scale_factor=1.0, output_stride=16):
    target_width, target_height = valid_resolution(
        source_img.shape[1] * scale_factor, source_img.shape[0] * scale_factor, output_stride=output_stride)
    scale = np.array([source_img.shape[0] / target_height, source_img.shape[1] / target_width])

    input_img = cv2.resize(source_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0
    input_img = input_img.transpose((2, 0, 1)).reshape(1, 3, target_height, target_width)
    return input_img, source_img, scale


def read_imgfile(path, scale_factor=1.0, output_stride=16):
    img = cv2.imread(path)
    return prepare_img(img, scale_factor, output_stride)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = posenet.load_model(101).to(device)

cam_stream = 'rtsp://192.168.0.142:7447/5ca3cbe70c0af3aa8fbeab20_1'

c = cv2.VideoCapture(cam_stream)

while True:
    ok, img = c.read()
    if not ok: break

    input_img, source_img, scale = prepare_img(img, scale_factor=1, output_stride=model.output_stride)
    input_img = torch.Tensor(input_img).to(device)


    


    
    model.eval()
    with torch.no_grad():
        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_img)


        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
            heatmaps_result.squeeze(0),
            offsets_result.squeeze(0),
            displacement_fwd_result.squeeze(0),
            displacement_bwd_result.squeeze(0),
            output_stride=model.output_stride,
            max_pose_detections=10,
            min_pose_score=0.25)

        img = posenet.draw_skel_and_kp(
                img, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.25, min_part_score=0.25)

        to_display(img)

        print('.')
    
