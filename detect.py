import os
import math
import random
import argparse
import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as mpcm
import matplotlib.patches as patches
import sys
sys.path.append('./')

from nets import ssd_vgg_512, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
#from notebooks import visualization




def bboxes_draw_on_img(img, classes, scores, bboxes, colors, thickness=2):
    shape = img.shape
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        color = [225,225,225]
        # Draw bounding box...
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        # Draw text...
        s = '%s/%.3f' % (classes[i], scores[i])
        p1 = (p1[0]-5, p1[1])
        #cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)


# =========================================================================== #
# Matplotlib show...
# =========================================================================== #
def plt_bboxes(img, classes, scores, bboxes, figsize=(10,10), linewidth=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=[random.random(), random.random(), random.random()],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            #class_name = str(cls_id)
            #plt.gca().text(xmin, ymin - 2,
                           #'{:s} | {:.3f}'.format(class_name, score),
                           #bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           #fontsize=12, color='white')
    plt.show()



# =========================================================================== #
# Some colormaps.
# =========================================================================== #
def colors_subselect(colors, num_classes=21):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i*dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors

colors_plasma = colors_subselect(mpcm.plasma.colors, num_classes=21)
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

def detect(args):
    # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    isess = tf.InteractiveSession(config=config)

    # Input placeholder.
    net_shape = (512,512)
    data_format = 'NHWC'
    img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
    # Evaluation pre-processing: resize to SSD net shape.
    image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
        img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
    image_4d = tf.expand_dims(image_pre, 0)

    # Define the SSD model.
    reuse = True if 'ssd_net' in locals() else None
    ssd_net = ssd_vgg_512.SSDNet()
    with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
        predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=None)
    
    # Restore SSD model.
    ckpt_filename = args.ckpt_filename
    #ckpt_filename = './tfmodel/model.ckpt-3080'
    isess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(isess, ckpt_filename)

    # SSD default anchor boxes.
    ssd_anchors = ssd_net.anchors(net_shape)

    for img_name in os.listdir(args.test_img_folder): 
        #args.test_img_folder
        img = mpimg.imread(os.path.join(args.test_img_folder, img_name))
        rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
        select_threshold=0.70
        nms_threshold=.1
         #Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                rpredictions, rlocalisations, ssd_anchors,
                select_threshold=select_threshold, img_shape=net_shape, num_classes=2, decode=True)
    
        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
          
        #Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        print(rclasses, rscores, rbboxes,img_name)
        bboxes_draw_on_img(img, rclasses, rscores, rbboxes, colors_plasma, 2)
        plt_bboxes(img, rclasses, rscores, rbboxes)

       


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect images using TensorFlow SSD network')
    parser.add_argument('-c', dest='ckpt_filename', 
            help='checkpoint file path',
            required=True, type=str)
    parser.add_argument('-d', dest='test_img_folder', 
            help='The images folder to be detected',
            required=True, type=str)
    parser.add_argument('--output', dest='output_fn',  
            help='The output path', 
            default='detect_results.txt', type=str) 

    args = parser.parse_args()
    detect(args)

