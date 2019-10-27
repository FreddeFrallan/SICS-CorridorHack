import tensorflow as tf
from MultiStreamModel import MultiStreamModel
import numpy as np
import cv2, time

from MultiStreamModel.ImageDetection.utils import label_map_util
from MultiStreamModel.ImageDetection.utils import visualization_utils_color as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


class TensoflowFaceDector(object):
    def __init__(self, PATH_TO_CKPT):
        """Tensorflow detector
        """

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True

    def run(self, image):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time
        #print('inference time cost: {}'.format(elapsed_time))

        return (boxes, scores, classes, num_detections)


def createOracle():
    model = TensoflowFaceDector(PATH_TO_CKPT)
    return model.run

def addBoxToImage(image, content):
    image = image.copy()
    try:
        (boxes, scores, classes, num_detections) = content
    except:
        return image

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4)

    return image

if (__name__ == '__main__'):
    streams = [

        # Sky News Live Stream
        ('localhost', 5001, 'https://www.youtube.com/watch?v=lrX6ktLg8WQ', 0),

        # 4 Fusion Breakthroughs | Answers With Joe
        #('localhost', 5001, "https://www.youtube.com/watch?v=lRa798QVEFQ", 0),

        ('localhost', 5001, "/home/fredrik/Documents/Superbowl.mp4", 2),

        # AlphaStar: The inside story (5 min)
        ('localhost', 5001, "https://www.youtube.com/watch?v=UuhECwm31dM", 4),

        # Esport interview with TSM Zven
        ('localhost', 5001, "https://www.youtube.com/watch?v=-hZJpgN6cF8", 6),

        # George Dyson - Computer History
        #('localhost', 5001, "https://www.youtube.com/watch?v=4si_FSCfe1g", 6),

        # camera
        #('localhost', 5001, 'rtsp://192.168.0.142:7447/5ca3cbe70c0af3aa8fbeab20_1', 6),

        # Train Halmstad to Gothenburg
        #('localhost', 5001, 'https://www.youtube.com/watch?v=1Rq9b_bn6Bc', 6),


        ('localhost', 5001, "/home/fredrik/Pictures/DemoPoster.png", 7),
        #('localhost', 5001, "https://www.youtube.com/watch?v=OOG65rSM5fA", 7),
        #('localhost', 5001, "https://www.youtube.com/watch?v=007VM8NZxkI", 7),

        #    ('localhost', 5001, "https://www.youtube.com/watch?v=vhACO_m5pH0", 4)
    ]

    MultiStreamModel.startStreams(createOracle, addBoxToImage, streams, "1")
