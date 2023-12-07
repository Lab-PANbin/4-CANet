import keras
import models
from utils.image import read_image_bgr, preprocess_image, resize_image
from utils.visualization import draw_box, draw_caption
from utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import glob
import os.path as osp
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.set_session(get_session())
# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = 'snapshots/2019-11-25/beifenresnet101/resnet101_pascal_75.h5'
print(model_path)
# load retinanet model
model = models.load_model(model_path, backbone_name='resnet101')
model = models.convert_model(model)
voc_classes = {
    'airplane': 0,
    'ship': 1,
    'storage tank': 2,
    'baseball diamond': 3,
    'tennis court': 4,
    'basketball court': 5,
    'ground track field': 6,
    'habor': 7,
    'bridge': 8,
    'vehicle': 9
}
labels_to_names = {}
for key, value in voc_classes.items():
    labels_to_names[value] = key
# load image
image_paths = glob.glob('test/*.jpg')
for image_path in image_paths:
    image = read_image_bgr(image_path)
    # copy to draw on
    draw = image.copy()
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)
    # process image
    start = time.time()
    # locations, feature_shapes = model.predict_on_batch(np.expand_dims(image, axis=0))
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    # print(type(model.predict_on_batch(np.expand_dims(image, axis=0))))
    # print(model.predict_on_batch(np.expand_dims(image, axis=0))[0].shape)
    print(image_path)
    print("processing time: ", time.time() - start)
    # correct for image scale
    boxes /= scale
    labels_to_locations = {}
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break
        start_x = int(box[0])
        start_y = int(box[1])
        end_x = int(box[2])
        end_y = int(box[3])
        color = label_color(label)
        b = box.astype(int)
        draw_box(draw, b, color=color)
        # caption = "{} {:.3f}".format(labels_to_names[label], score)
        caption = "{}".format(labels_to_names[label])
        draw_caption(draw, b, caption)
    draw_box(draw, b, color=color)
    print('testresult/'+image_path+'has been precessed')
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image', draw)
    cv2.imwrite('testresult'+'/'+'test30'+'/'+image_path, draw)
    key = cv2.waitKey(0)
    if int(key) == 121:
        image_fname = osp.split(image_path)[-1]
        cv2.imwrite('test/{}'.format(image_fname), draw)

