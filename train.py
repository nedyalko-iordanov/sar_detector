#! /usr/bin/env python

import argparse
import os
import numpy as np
import json
from voc import parse_voc_annotation
from yolo import create_yolov3_model, dummy_loss
from generator import BatchGenerator
from utils.utils import normalize, evaluate, makedirs
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from callbacks import CustomModelCheckpoint, CustomTensorBoard
from utils.multi_gpu_model import multi_gpu_model
import tensorflow as tf
import keras
from keras.models import load_model
from random_generator import ImageGenerator
from generator_2 import RandomBatchGenerator
from random_generator_single_image import ImageGeneratorSingle

config = tf.compat.v1.ConfigProto(
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def create_training_instances(
    train_annot_folder,
    train_image_folder,
    train_cache,
    valid_annot_folder,
    valid_image_folder,
    valid_cache,
    labels,
):
    # parse annotations of the training set
    train_ints, train_labels = parse_voc_annotation(train_annot_folder, train_image_folder, train_cache, labels)

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(valid_annot_folder):
        valid_ints, valid_labels = parse_voc_annotation(valid_annot_folder, valid_image_folder, valid_cache, labels)
    else:
        print("valid_annot_folder not exists. Spliting the trainining set.")

        train_valid_split = int(0.8*len(train_ints))
        np.random.seed(0)
        np.random.shuffle(train_ints)
        np.random.seed()

        valid_ints = train_ints[train_valid_split:]
        train_ints = train_ints[:train_valid_split]

    # compare the seen labels with the given labels in config.json
    if len(labels) > 0:
        overlap_labels = set(labels).intersection(set(train_labels.keys()))

        print('Seen labels: \t'  + str(train_labels) + '\n')
        print('Given labels: \t' + str(labels))

        # return None, None, None if some given label is not in the dataset
        if len(overlap_labels) < len(labels):
            print('Some labels have no annotations! Please revise the list of labels in the config.json.')
            return None, None, None
    else:
        print('No labels are provided. Train on all seen labels.')
        print(train_labels)
        labels = train_labels.keys()

    max_box_per_image = max([len(inst['object']) for inst in (train_ints + valid_ints)])

    return train_ints, valid_ints, sorted(labels), max_box_per_image

def create_callbacks(saved_weights_name, tensorboard_logs, model_to_save):
    makedirs(tensorboard_logs)
    
    early_stop = EarlyStopping(
        monitor     = 'loss', 
        min_delta   = 0.01, 
        patience    = 7, 
        mode        = 'min',
        verbose     = 1
    )
    checkpoint = CustomModelCheckpoint(
        model_to_save   = model_to_save,
        filepath        = saved_weights_name + '{epoch:02d}.h5',
        monitor         = 'loss', 
        verbose         = 1, 
        save_best_only  = False,
        mode            = 'min', 
        period          = 1
    )
    reduce_on_plateau = ReduceLROnPlateau(
        monitor  = 'loss',
        factor   = 0.1,
        patience = 2,
        verbose  = 1,
        mode     = 'min',
        epsilon  = 0.01,
        cooldown = 0,
        min_lr   = 0
    )
    tensorboard = CustomTensorBoard(
        log_dir                = tensorboard_logs,
        write_graph            = True,
        write_images           = True,
    )    
    return [early_stop, checkpoint, reduce_on_plateau, tensorboard]

def create_model(
    nb_class, 
    anchors, 
    max_box_per_image, 
    max_grid, batch_size, 
    warmup_batches, 
    ignore_thresh, 
    multi_gpu, 
    saved_weights_name,
    pretrained_weights_location,
    lr,
    grid_scales,
    obj_scale,
    noobj_scale,
    xywh_scale,
    class_scale,
    logdir
):
    if multi_gpu > 1:
        with tf.device('/cpu:0'):
            template_model, infer_model = create_yolov3_model(
                nb_class            = nb_class, 
                anchors             = anchors, 
                max_box_per_image   = max_box_per_image, 
                max_grid            = max_grid, 
                batch_size          = batch_size//multi_gpu, 
                warmup_batches      = warmup_batches,
                ignore_thresh       = ignore_thresh,
                grid_scales         = grid_scales,
                obj_scale           = obj_scale,
                noobj_scale         = noobj_scale,
                xywh_scale          = xywh_scale,
                class_scale         = class_scale,
                logdir              = logdir
            )
    else:
        template_model, infer_model = create_yolov3_model(
            nb_class            = nb_class, 
            anchors             = anchors, 
            max_box_per_image   = max_box_per_image, 
            max_grid            = max_grid, 
            batch_size          = batch_size, 
            warmup_batches      = warmup_batches,
            ignore_thresh       = ignore_thresh,
            grid_scales         = grid_scales,
            obj_scale           = obj_scale,
            noobj_scale         = noobj_scale,
            xywh_scale          = xywh_scale,
            class_scale         = class_scale,
            logdir              = logdir
        )  

    # load the pretrained weight if exists, otherwise load the backend weight only
    if os.path.exists(saved_weights_name): 
        print("\nLoading pretrained weights.\n")
        template_model.load_weights(saved_weights_name)
    else:
        template_model.load_weights(pretrained_weights_location, by_name=True)

    if multi_gpu > 1:
        train_model = multi_gpu_model(template_model, gpus=multi_gpu)
    else:
        train_model = template_model      

    optimizer = Adam(lr=lr, clipnorm=0.001)

    # Freeze layers
    pred_layer_idx = [80, 81, #-yolo1
                      92, 93, #-yolo2
                      104, 105] #-yolo3
    for layer in train_model.layers:
        if int(layer.name.split('_')[-1]) not in pred_layer_idx:
            layer.trainable = False
            print('Layer ' + layer.name + ' is set to NOT trainable.')
        else:
            print('Layer ' + layer.name + ' is set to trainable.')

    train_model.compile(loss=dummy_loss, optimizer=optimizer)             

    return train_model, infer_model

def _main_(args):
    #config_path = args.get('conf')
    #images_location = args.get('images_location')
    #background_images_location = args.get('background_images_location')
    #saved_weights_location = args.get('saved_weights_location')
    #pretrained_weights_location = args.get('pretrained_weights_location')



    config_path = args.conf
    images_location = args.articles
    background_images_location = args.background
    saved_weights_location = args.saveloc
    pretrained_weights_location = args.pretrained

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    labels = config['model']['labels']

    ###############################
    #   Parse the annotations 
    ###############################
    #train_ints, valid_ints, labels, max_box_per_image = create_training_instances(
    #    config['train']['train_annot_folder'],
    #    config['train']['train_image_folder'],
    #    config['train']['cache_name'],
    #    config['valid']['valid_annot_folder'],
    #    config['valid']['valid_image_folder'],
    #    config['valid']['cache_name'],
    #    config['model']['labels']
    #)
    #print('\nTraining on: \t' + str(labels) + '\n')

    ###############################
    #   Create the generators 
    ###############################
    #train_generator = BatchGenerator(
    #    instances           = train_ints,
    #    anchors             = config['model']['anchors'],
    #    labels              = labels,
    #    downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
    #    max_box_per_image   = max_box_per_image,
    #    batch_size          = config['train']['batch_size'],
    #    min_net_size        = config['model']['min_input_size'],
    #    max_net_size        = config['model']['max_input_size'],
    #    shuffle             = True,
    #    jitter              = 0.3,
    #    norm                = normalize
    #)


    random_generator = ImageGeneratorSingle(images_location,
                               background_images_location,
                               object_proportion_range=[0.4, 0.9],
                               max_perspective_jitter=0.1,
                               max_object_overlap_area=0.2)
    train_generator = RandomBatchGenerator(
        random_generator    = random_generator,
        labels              = labels,
        epoch_size          = config['train']['epoch_size'],
        anchors             = config['model']['anchors'],
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],
        norm                = normalize
    )

    #batch = train_generator.__getitem__(1)
    #valid_generator = BatchGenerator(
    #    instances           = valid_ints,
    #    anchors             = config['model']['anchors'],
    #    labels              = labels,
    #    downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
    #    max_box_per_image   = max_box_per_image,
    #    batch_size          = config['train']['batch_size'],
    #    min_net_size        = config['model']['min_input_size'],
    #    max_net_size        = config['model']['max_input_size'],
    #    shuffle             = True,
    #    jitter              = 0.0,
    #    norm                = normalize
    #)

    ###############################
    #   Create the model 
    ###############################
    if os.path.exists(saved_weights_location):
        config['train']['warmup_epochs'] = 0
    warmup_batches = config['train']['warmup_epochs'] * (config['train']['train_times']*len(train_generator))   

    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    multi_gpu = len(config['train']['gpus'].split(','))

    train_model, infer_model = create_model(
        nb_class            = len(labels),
        anchors             = config['model']['anchors'], 
        max_box_per_image   = train_generator.random_generator.max_objects,
        max_grid            = [config['model']['max_input_size'], config['model']['max_input_size']], 
        batch_size          = config['train']['batch_size'], 
        warmup_batches      = warmup_batches,
        ignore_thresh       = config['train']['ignore_thresh'],
        multi_gpu           = multi_gpu,
        saved_weights_name  = saved_weights_location,
        pretrained_weights_location = pretrained_weights_location,
        lr                  = config['train']['learning_rate'],
        grid_scales         = config['train']['grid_scales'],
        obj_scale           = config['train']['obj_scale'],
        noobj_scale         = config['train']['noobj_scale'],
        xywh_scale          = config['train']['xywh_scale'],
        class_scale         = config['train']['class_scale'],
        logdir              = config['train']['tensorboard_dir']
    )

    ###############################
    #   Kick off the training
    ###############################
    callbacks = create_callbacks(saved_weights_location, config['train']['tensorboard_dir'], infer_model)
    infer_model.save('infer_model.h5')

    train_model.fit_generator(
        generator        = train_generator, 
        steps_per_epoch  = len(train_generator) * config['train']['train_times'], 
        epochs           = config['train']['nb_epochs'] + config['train']['warmup_epochs'], 
        verbose          = 1,
        callbacks        = callbacks, 
        workers          = 8,
        max_queue_size   = 32
    )

    # make a GPU version of infer_model for evaluation
    if multi_gpu > 1:
        infer_model = load_model(saved_weights_location)

    ###############################
    #   Run the evaluation
    ###############################   
    # compute mAP for all the classes
    #average_precisions = evaluate(infer_model, valid_generator)

    # print the score
    #for label, average_precision in average_precisions.items():
    #    print(labels[label] + ': {:.4f}'.format(average_precision))
    #print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='train and evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-a', '--articles', help='path to article images with cropping paths')
    argparser.add_argument('-b', '--background', help='path to background images')
    argparser.add_argument('-s', '--saveloc', help='path to save model')
    argparser.add_argument('-p', '--pretrained', help='path to pretrained weights file')
    args = argparser.parse_args()

    #args = {
    #    'conf': 'C:\\Users\\Freeware Sys\\PycharmProjects\\sar_detector\\zoo\\config_detector_debug.json',
    #    'images_location': 'C:\\Users\\Freeware Sys\\Desktop\\articles\\raw_debug',
    #    'background_images_location': 'C:\\Users\\Freeware Sys\\Desktop\\articles\\backgrounds_debug',
    #    'saved_weights_location': 'C:\\Users\\Freeware Sys\\PycharmProjects\\sar_detector\\03_detector_',
    #    #'pretrained_weights_location': 'C:\\Users\\Freeware Sys\\PycharmProjects\\sar_detector\\detector_epoch_1.h5',
    #    'pretrained_weights_location': 'C:\\Users\\Freeware Sys\\PycharmProjects\\keras-yolo3\\backend.h5',
    #}
    _main_(args)
