import cv2
import copy
import numpy as np
from keras.utils import Sequence
from utils.bbox import BoundBox, bbox_iou
from utils.image import apply_random_scale_and_crop, random_distort_image, random_flip, correct_bounding_boxes
import time

class RandomBatchGenerator(Sequence):
    def __init__(self,
                 random_generator,
                 anchors,
                 epoch_size=1024,
                 downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
                 batch_size=1,
                 min_net_size=320,
                 max_net_size=608,
                 norm=None
                 ):
        self.random_generator = random_generator
        self.batch_size = batch_size
        self.labels = random_generator.labels
        self.epoch_size = epoch_size
        self.downsample = downsample
        self.max_box_per_image = random_generator.max_objects
        self.min_net_size = (min_net_size // self.downsample) * self.downsample
        self.max_net_size = (max_net_size // self.downsample) * self.downsample
        self.norm = norm
        self.anchors = [BoundBox(0, 0, anchors[2 * i], anchors[2 * i + 1]) for i in range(len(anchors) // 2)]
        self.net_h = 416
        self.net_w = 416
        self.epoch = 0

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        print('Generating batch: ' + str(idx))
        start = time.time()
        # get image input size, change every 10 batches
        net_h, net_w = self._get_net_size(idx)
        base_grid_h, base_grid_w = net_h // self.downsample, net_w // self.downsample


        x_batch = np.zeros((self.batch_size, net_h, net_w, 3))  # input images
        t_batch = np.zeros((self.batch_size, 1, 1, 1, self.max_box_per_image, 4))  # list of groundtruth boxes

        # initialize the inputs and the outputs
        yolo_1 = np.zeros((self.batch_size, 1 * base_grid_h, 1 * base_grid_w, len(self.anchors) // 3,
                           4 + 1 + len(self.labels)))  # desired network output 1
        yolo_2 = np.zeros((self.batch_size, 2 * base_grid_h, 2 * base_grid_w, len(self.anchors) // 3,
                           4 + 1 + len(self.labels)))  # desired network output 2
        yolo_3 = np.zeros((self.batch_size, 4 * base_grid_h, 4 * base_grid_w, len(self.anchors) // 3,
                           4 + 1 + len(self.labels)))  # desired network output 3
        yolos = [yolo_3, yolo_2, yolo_1]

        dummy_yolo_1 = np.zeros((self.batch_size, 1))
        dummy_yolo_2 = np.zeros((self.batch_size, 1))
        dummy_yolo_3 = np.zeros((self.batch_size, 1))

        instance_count = 0
        true_box_index = 0

        all_images, all_objects = self.random_generator.generate_random_batch(
            batch_size=self.batch_size,
            target_size=net_w,
            seed=idx*100+self.epoch
        )

        # do the logic to fill in the inputs and the output
        for img, all_objs in zip(all_images, all_objects):

            for obj in all_objs:
                # find the best anchor box for this object
                max_anchor = None
                max_index = -1
                max_iou = -1

                shifted_box = BoundBox(0,
                                       0,
                                       obj['xmax'] - obj['xmin'],
                                       obj['ymax'] - obj['ymin'])

                for i in range(len(self.anchors)):
                    anchor = self.anchors[i]
                    iou = bbox_iou(shifted_box, anchor)

                    if max_iou < iou:
                        max_anchor = anchor
                        max_index = i
                        max_iou = iou

                        # determine the yolo to be responsible for this bounding box
                yolo = yolos[max_index // 3]
                grid_h, grid_w = yolo.shape[1:3]

                # determine the position of the bounding box on the grid
                center_x = .5 * (obj['xmin'] + obj['xmax'])
                center_x = center_x / float(net_w) * grid_w  # sigma(t_x) + c_x
                center_y = .5 * (obj['ymin'] + obj['ymax'])
                center_y = center_y / float(net_h) * grid_h  # sigma(t_y) + c_y

                # determine the sizes of the bounding box
                w = np.log((obj['xmax'] - obj['xmin']) / float(max_anchor.xmax))  # t_w
                h = np.log((obj['ymax'] - obj['ymin']) / float(max_anchor.ymax))  # t_h

                box = [center_x, center_y, w, h]

                # determine the index of the label
                obj_indx = self.labels.index(obj['name'])

                # determine the location of the cell responsible for this object
                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                # assign ground truth x, y, w, h, confidence and class probs to y_batch
                yolo[instance_count, grid_y, grid_x, max_index % 3] = 0
                yolo[instance_count, grid_y, grid_x, max_index % 3, 0:4] = box
                yolo[instance_count, grid_y, grid_x, max_index % 3, 4] = 1.
                yolo[instance_count, grid_y, grid_x, max_index % 3, 5 + obj_indx] = 1

                # assign the true box to t_batch
                true_box = [center_x, center_y, obj['xmax'] - obj['xmin'], obj['ymax'] - obj['ymin']]
                t_batch[instance_count, 0, 0, 0, true_box_index] = true_box

                true_box_index += 1
                true_box_index = true_box_index % self.max_box_per_image

                # assign input image to x_batch
            if self.norm != None:
                x_batch[instance_count] = self.norm(img)
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    cv2.rectangle(img, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), (255, 0, 0), 3)
                    cv2.putText(img, obj['name'],
                                (obj['xmin'] + 2, obj['ymin'] + 12),
                                0, 1.2e-3 * img.shape[0],
                                (0, 255, 0), 2)

                x_batch[instance_count] = img

            # increase instance counter in the current batch
            instance_count += 1
        seconds = time.time() - start
        print('Batch generated for ' + str(round(seconds, 2)) + ' sec.')
        return [x_batch, t_batch, yolo_1, yolo_2, yolo_3], [dummy_yolo_1, dummy_yolo_2, dummy_yolo_3]

    def _get_net_size(self, idx):
        if idx % 10 == 0:
            net_size = self.downsample * np.random.randint(self.min_net_size / self.downsample, \
                                                           self.max_net_size / self.downsample + 1)
            print("resizing: ", net_size, net_size)
            self.net_h, self.net_w = net_size, net_size
        return self.net_h, self.net_w

    def on_epoch_end(self):
        self.epoch += 1

    def num_classes(self):
        return len(self.labels)

    def size(self):
        return self.epoch_size * self.batch_size

    def get_anchors(self):
        anchors = []

        for anchor in self.anchors:
            anchors += [anchor.xmax, anchor.ymax]

        return anchors