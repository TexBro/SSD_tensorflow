#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 05:45:54 2017

@author: donghyun
"""
import os
import argparse
import numpy as np
import tensorflow as tf
import cv2
import pickle
import math
from time import time

slim = tf.contrib.slim

#from PIL import Image

#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import matplotlib.cm as mpcm
#import matplotlib.patches as patches

import sys	
sys.path.append('./')
from nets import ssd_vgg_512, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
#from notebooks import visualization

net_shape=(512,512)
select_threshold=0.6
nms_threshold=.01

def FPS(prev_time,num_of_img):
    
    cur_time = time()
    total_time=cur_time-prev_time
    fps=num_of_img/total_time
    return fps,total_time
    

def adjust_bboxes(height,width,x1,y1,bboxes):

    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        p1 = (int(bbox[0] * net_shape[0]), int(bbox[1] * net_shape[1]))
        p2 = (int(bbox[2] * net_shape[0]), int(bbox[3] * net_shape[1]))
        
        adjust_p1=(float(p1[0]+y1)/float(height),float(p1[1]+x1)/float(width))
        adjust_p2=(float(p2[0]+y1)/float(height),float(p2[1]+x1)/float(width))
             
        bboxes[i]=list(adjust_p1+adjust_p2)
        
    return bboxes
    
def bboxes_draw_on_img(img, classes, scores, bboxes, thickness=2):
    shape = img.shape
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        color = [0,0,255]
        # Draw bounding box...
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        # Draw text...
        s = '%s/%.3f' % ('crack', scores[i])
        p1 = (p1[0]-5, p1[1])
        cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)


def img_crop_bounding(height,width,x1_,y1_):
    x1=x1_*net_shape[1]
    y1=y1_*net_shape[0]
    #check edges when cropping
    if x1+net_shape[1]>width and y1+net_shape[0]>height:
        crop_width = width - x1
        crop_height= height - y1
    
    elif y1+net_shape[0]>height:
        crop_height = height-y1
        crop_width=net_shape[1]
    
    elif x1+net_shape[1]>width:
        crop_width = width - x1
        crop_height=net_shape[0]
    
    else:
        crop_height=net_shape[0]
        crop_width=net_shape[1]
    
    return x1,y1,crop_height,crop_width

def detect_img(img,isess,image_4d,predictions,localisations,bbox_img,img_input,ssd_anchors):

    cap = cv2.VideoCapture(img)
    while(cap.isOpened()):
       
        re, frame = cap.read()
       
        if re == False:
            return -1
        
        org_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height,width = org_img.shape[:2]
        merge_img=np.ones((height,width,3),np.uint8)
        merge_bboxes=[]
        merge_scores=[]
        x=int(math.ceil(width/512.))
        y=int(math.ceil(height/512.))
        detected=0
        for y1_ in range(0,y):
            for x1_ in range(0,x):
        
                x1,y1,crop_height,crop_width=img_crop_bounding(height,width,x1_,y1_)

                print(height,width,crop_height,crop_width,y1,x1)
        
                img= org_img[y1:y1+crop_height,x1:x1+crop_width]
                
                rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                                      feed_dict={img_input: img})

            # Get classes and bboxes from the net outputs.
                rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                        rpredictions, rlocalisations, ssd_anchors,
                        select_threshold=select_threshold, img_shape=net_shape, num_classes=2, decode=True)
        
                rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
                rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
                rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
            # Resize bboxes to original image shape. Note: useless for Resize.WARP!
                rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    #               cv2.imshow('frame',img)
                bboxes_draw_on_img(img, rclasses, rscores, rbboxes, 2)
                
                merge_img[y1:y1+crop_height,x1:x1+crop_width]=img
                
                if len(rbboxes):                
                    merge_bboxes.append(adjust_bboxes(height,width,x1,y1,rbboxes))
                    merge_scores.append(rscores)
                    detected=1
                else:
                    merge_bboxes.append(np.zeros((0,4),dtype=np.float32))
                    merge_scores.append(np.zeros((0),dtype=np.float32))
                

                #cv2.imwrite('./test'+str(i)+'.jpg',merge_img)
        #cv2.waitKey(1000)
        img_num_tuple=(x,y)
        return merge_img, merge_bboxes, merge_scores, img_num_tuple, detected


def detect(args):
    # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    isess = tf.InteractiveSession(config=config)

    # Input placeholder.
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

    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    for img_name in os.listdir(args.test_img_folder): #args.test_img_folder
        img = os.path.join(args.test_img_folder, img_name)

        prev_time=time()
        
        merge_img,boxes, scores ,num_of_img,detected= detect_img(img,isess,image_4d,predictions,localisations,bbox_img,img_input,ssd_anchors)        
        
        f1= open('bboxes.txt','wb')
        pickle.dump(boxes,f1)
        f2= open('scores.txt','wb')
        pickle.dump(scores,f2)            
        
        fps,total_time=FPS(prev_time,num_of_img[0]*num_of_img[1])
        print("FPS :  %f " %fps," total time : %f "%total_time )
        print(num_of_img)        
        cv2.resizeWindow('frame', 5000,5000)
        cv2.imshow('frame',merge_img)
        cv2.waitKey(10000)
              
        print(boxes)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect images using TensorFlow SSD network')
    parser.add_argument('-c', dest='ckpt_filename', default='./model.ckpt-30412',
            help='checkpoint file path',
            type=str)
    parser.add_argument('-d', dest='test_img_folder', 
            help='The images folder to be detected',
            default='./test4', type=str)
    parser.add_argument('--output', dest='output_fn',  
            help='The output path', 
            default='detect_results.txt', type=str) 

    args = parser.parse_args()
    detect(args)

