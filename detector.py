#! /usr/bin/env python

import os
import argparse
import json
import cv2
from scipy.special import expit
#from utils.utils import makedirs, correct_yolo_boxes, decode_netout, bbox_iou
#from utils.bbox import draw_boxes, draw_boxes_detection_only
from keras.models import load_model
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt


#utilities

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c       = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin

    union = w1*h1 + w2*h2 - intersect

    return float(intersect) / union


def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_w = net_w
        new_h = (image_h*net_w)/image_w
    else:
        new_h = net_w
        new_w = (image_w*net_h)/image_h

    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h

        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


def _sigmoid(x):
    return expit(x)

def _softmax(x, axis=-1):
    x = x - np.amax(x, axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / e_x.sum(axis, keepdims=True)

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5

    boxes = []

    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4]   = _sigmoid(netout[..., 4])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h*grid_w):
        row = i // grid_w
        col = i % grid_w

        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[row, col, b, 4]

            if(objectness <= obj_thresh): continue

            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[row,col,b,:4]

            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height

            # last elements are class probabilities
            classes = netout[row,col,b,5:]

            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)

            boxes.append(box)

    return boxes


class Detector:
    def __init__(self,
                 pretrained_weights_path,
                 net_size=416,
                 obj_thresh=0.5,
                 nms_thresh=0.45,
                 anchors=[17,18, 28,24, 36,34, 42,44, 56,51, 72,66, 90,95, 92,154, 139,281],
                 output_path=None):
        self.pretrained_weights_path = pretrained_weights_path
        self.output_path = output_path
        self.net_size = net_size
        self.obj_thresh = obj_thresh
        self.nms_thresh = nms_thresh
        self.infer_model = load_model(self.pretrained_weights_path)
        self.anchors = anchors
        self.called = 0

    def crop_center(self, image):
        height, width, _ = image.shape
        min_size = min(height, width)
        h_idx = (height - min_size) // 2
        w_idx = (width - min_size) // 2
        image = image[h_idx:h_idx+min_size, w_idx:w_idx+min_size]
        return image

    def do_nms(self, boxes, nms_thresh):
        good_boxes = []
        if len(boxes) > 0:
            # sort indices on their objectness score if it's bigger than obj_tresh
            sorted_indices = np.argsort([-box.c for box in boxes if box.c >= self.obj_thresh])
            # loop over each index and suppress predictions to be 0 for overlapping boxes with lower objectness
            for i, index in enumerate(sorted_indices):
                if boxes[index].c == 0:
                    continue
                good_boxes += [boxes[index]]
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if bbox_iou(boxes[index], boxes[index_j]) >= nms_thresh:
                        boxes[index_j].c = 0
        return good_boxes

    def infer(self, image, save=False):
        self.called += 1
        # crop center square
        image = self.crop_center(image)
        image_h, image_w, _ = image.shape
        #expand dims for inference
        batch_input = np.expand_dims(
            cv2.resize(image/255., # normalize 0 to 1
                       (self.net_size, self.net_size)) # resize to net size
            , axis=0)

        # run the prediction
        batch_output = self.infer_model.predict_on_batch(batch_input)

        # get the yolo predictions
        yolos = [batch_output[0][0], batch_output[1][0], batch_output[2][0]]
        boxes = []

        # decode the output of the network
        for j in range(len(yolos)):
            yolo_anchors = self.anchors[(2-j)*6:(3-j)*6] # config['model']['anchors']
            boxes += decode_netout(yolos[j], yolo_anchors, self.obj_thresh, self.net_size, self.net_size)

        # correct the sizes of the bounding boxes
        correct_yolo_boxes(boxes, image_h, image_w, self.net_size, self.net_size)

        # suppress non-maximal boxes
        boxes = self.do_nms(boxes, self.nms_thresh)

        if len(boxes) == 0:
            cropped = None # return None if no boxes are detected
        else:
            biggest_box = max(boxes, key=lambda x: (x.xmax - x.xmin)*(x.ymax - x.ymin) if (x.c > 0) else 0)
            xmin = max(biggest_box.xmin, 0)
            ymin = max(biggest_box.ymin, 0)
            xmax = min(biggest_box.xmax, image_w)
            ymax = min(biggest_box.ymax, image_h)
            cropped = image[ymin:ymax, xmin:xmax]
        if save:
            draw_boxes_detection_only(image, boxes, None, self.obj_thresh)
            cv2.imwrite(os.path.join(self.output_path, str(self.called) + '.jpg'), image)
        return cropped

if __name__ == '__main__':
    detector = Detector(pretrained_weights_path="/Users/yordanovn/Downloads/1_detector_05 (1).h5",
                        output_path="/Users/yordanovn/Downloads/test_images/test_images_phone/detected_7/")

    image = "/Users/yordanovn/Downloads/test_images/test_images_phone/1639411806957.jpg"
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('cropped.png', detector.infer(image, save=False))
