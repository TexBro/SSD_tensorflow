# SSD: Single Shot MultiBox Detector in TensorFlow

SSD is an unified framework for object detection with a single network. It has been originally introduced in this research [article](http://arxiv.org/abs/1512.02325).

This repository contains a TensorFlow re-implementation of the original [Caffe code](https://github.com/weiliu89/caffe/tree/ssd). At present, it only implements VGG-based SSD networks (with 300 and 512 inputs), but the architecture of the project is modular, and should make easy the implementation and training of other SSD variants (ResNet or Inception based for instance). Present TF checkpoints have been directly converted from SSD Caffe models.

The organisation is inspired by the TF-Slim models repository containing the implementation of popular architectures (ResNet, Inception and VGG). Hence, it is separated in three main parts:
* datasets: interface to popular datasets (Pascal VOC, COCO, ...) and scripts to convert the former to TF-Records;
* networks: definition of SSD networks, and common encoding and decoding methods (we refer to the paper on this precise topic);
* pre-processing: pre-processing and data augmentation routines, inspired by original VGG and Inception implementations.

[![Alt text for your video](https://youtu.be/b56m1uQlySg/0.jpg)](https://youtu.be/b56m1uQlySg)

=========================================================================== #
# Fine tune VGG-based SSD network
# =========================================================================== #

DATASET_DIR=./result
TRAIN_DIR=./log_finetune/
CHECKPOINT_PATH=./log/model.ckpt-7959
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=train \
    --model_name=ssd_512_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=ssd_512_vgg \
    --save_summaries_secs=6 \sou
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.00001 \
    --learning_rate_decay_factor=0.94 \
    --batch_size=16
    
    
# =========================================================================== #
# Test VGG-based SSD network
# =========================================================================== #
EVAL_DIR=./logs/
CHECKPOINT_PATH=./log_finetune/model.ckpt-7087

python eval_ssd_network.py \    
    --eval_dir=${EVAL_DIR} \    
    --dataset_dir=./result \    
    --dataset_name=pascalvoc_2007 \    
    --dataset_split_name=test \    
    --model_name=ssd_512_vgg \    
    --checkpoint_path=${CHECKPOINT_PATH} \    
    --batch_size=1

