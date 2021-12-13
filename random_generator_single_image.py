import numpy as np
import cv2
import json
from matplotlib import pyplot as plt
import math
import os
import shutil
from random_generator import *

class ImageGeneratorSingle:
    def __init__(
            self,
            article_images_path,
            background_images_path,
            output_resolution_range=[352, 448],  # taken from https://github.com/experiencor/keras-yolo3 config file
            object_proportion_range=[0.1, 0.4],
            min_visible_object_area=0.6,
            max_object_overlap_area=0.4,
            max_perspective_jitter=0.2,
            luminocity_range=[0.5, 1.5]
    ):
        """
        :param article_images_path: directory where images of article are stored with subfolders
            for each article and jsons with segmentation paths for each image
        :param background_images_path: directory where background images are stored
        :param output_resolution_range: list of min_value, max_value that will be the output resolution of the generator.
            Each batch will be produced with a random range
        :param objects_per_image_range: list of min_value, max_value that will be the number of articles in a single generated
            image.
        :param object_proportion_range: list of min_value, max_value that will be article proportion relative to the backgrond image
        :param min_visible_object_area: minimum proportion of object area that has to be within the boundaries of the image
        :param max_oject_overlap_area: maximum overlapping between to images allowed
        :param max_perspective_jitter: how much can each edgepoint vary proportinally to the article size to
            create perspective transformations
        :param luminocity_range: list of min_value, max_value that will be the luminocity adjustment for the whole image
        """
        self.article_images_path = article_images_path
        self.background_images_path = background_images_path
        self.min_output_resolution, self.max_output_resolution = output_resolution_range
        self.min_object_proportion, self.max_object_proportion = object_proportion_range
        self.min_visible_object_area = min_visible_object_area
        self.max_overlap_area = max_object_overlap_area
        self.max_perspective_jitter = max_perspective_jitter
        self.min_luminocity, self.max_luminocity = luminocity_range
        self.article_images, self.labels = self.get_cropped_article_images()
        self.backgrounds = self.get_backgrounds()
        self.max_objects = 1

    def get_cropped_article_images(self):
        articles = {}
        labels = []
         # iterate over all articles in the location, read their annotations, crop, augment and save in the database location
        for article_id in [x for x in os.listdir(self.article_images_path) if
                           os.path.isdir(os.path.join(self.article_images_path, x))]:
            labels += [article_id]
            images = []
            current_directory = os.path.join(self.article_images_path, article_id)
            jsons = [x for x in os.listdir(current_directory) if
                     (('json' in x) and (os.path.isfile(os.path.join(current_directory, x))))]
            for json_file in jsons:
                print(json_file)
                with open(os.path.join(current_directory, json_file)) as f:
                    image_annotations = json.load(f)
                points = image_annotations['shapes'][0]['points']
                points = np.array(points, dtype=np.int32)
                source = os.path.join(current_directory, image_annotations['imagePath'])
                image = np.uint8(cv2.imread(source))
                image = crop_object(image, points)
                images += [image]
            articles.update({article_id: images})
        return articles, labels

    def get_backgrounds(self):
        backgrounds = []
        images = [x for x in os.listdir(self.background_images_path) if ('jpeg' in x) or ('png' in x) or ('jpg' in x)]
        images = [os.path.join(self.background_images_path, x) for x in images]
        for image in images:
            img = cv2.imread(image)
            backgrounds += [img]
        return backgrounds

    def get_random_background(self):
        image = np.random.choice(self.backgrounds)
        size = min(image.shape[:-1])
        rotation_angle = np.random.randint(0, 359)
        image = rotate_image(image, rotation_angle)
        height, width, _ = image.shape
        bl = image[:int(height/2), :int(width/2)]
        br = image[:int(height/2), int(width/2):]
        tl = image[int(height/2):, :int(width/2)]
        tr = image[int(height/2):, :int(width/2):]
        non_black_bl = np.argwhere(np.max(bl != np.array([0, 0, 0]), axis=2))
        non_black_br = np.argwhere(np.max(br != np.array([0, 0, 0]), axis=2))
        non_black_tl = np.argwhere(np.max(tl != np.array([0, 0, 0]), axis=2))
        non_black_tr = np.argwhere(np.max(tr != np.array([0, 0, 0]), axis=2))
        random_bl = non_black_bl[np.random.choice(range(len(non_black_bl)))] + np.array([0, 0])
        random_br = non_black_br[np.random.choice(range(len(non_black_br)))] + np.array([0, width/2])
        random_tl = non_black_tl[np.random.choice(range(len(non_black_tl)))] + np.array([height/2, 0])
        random_tr = non_black_tr[np.random.choice(range(len(non_black_tr)))] + np.array([height/2, width/2])
        src_pnts = np.array([
            np.flip(random_bl),
            np.flip(random_tl),
            np.flip(random_tr),
            np.flip(random_br),
        ]).astype(np.float32)
        dst_pnts = np.array([
            [0, 0],
            [0, size-1],
            [size - 1, size - 1],
            [size-1, 0],
        ]).astype(np.float32)
        M = cv2.getPerspectiveTransform(src_pnts, dst_pnts)
        image = cv2.warpPerspective(image, M, (size, size))
        return image

    def get_random_article_photo(self, article):
        image_ind = np.random.choice(range(len(self.article_images[article])))
        image = self.article_images[article][image_ind]
        height, width, _ = image.shape
        rotation_angle = np.random.randint(0, 359)
        image = apply_random_perspective(image, self.max_perspective_jitter)
        image = rotate_image(image, rotation_angle)
        image = crop_black_pixels(image)
        return image

    def generate_random_image(self, target_size=448):
        article_label = np.random.choice(list(self.article_images.keys()), 1, replace=True)[0]
        article = self.get_random_article_photo(article_label)
        background = self.get_random_background()
        background_height, background_width, _ = background.shape
        # resize article to target scale
        height, width, _ = article.shape
        scale = np.random.uniform(self.min_object_proportion, self.max_object_proportion)
        target_max_size = max(int(background_height * scale), int(background_width * scale))
        article, _ = resize_to_max_size(article, target_max_size)
        height, width, _ = article.shape
        rest_height = (height % 2) ^ 1
        rest_width = (width % 2) ^ 1
        half_height = int(height/2)
        half_width = int(width/2)
        article_center_y = np.random.randint(half_height + 1, background_height - 1 - half_height)
        article_center_x = np.random.randint(half_width + 1, background_width - 1 - half_width)
        article_bl = [article_center_y - half_height, article_center_x - half_width]
        article_tl = [article_center_y + half_height, article_center_x - half_width]
        article_tr = [article_center_y + half_height, article_center_x + half_width]
        article_br = [article_center_y - half_height, article_center_x + half_width]
        bbox = np.array([
            article_bl,
            article_tl,
            article_tr,
            article_br
        ]).astype(np.int32)
        padded = np.pad(article, ((article_bl[0] + rest_height, background_height - 1 - article_tr[0]),
                                  (article_bl[1] + rest_width, background_width - 1 - article_tr[1]),
                                  (0, 0)))
        inv_mask = np.min(padded == np.array([0, 0, 0]), axis=2).astype(np.int8)
        background = cv2.bitwise_and(background, background,  mask=inv_mask)
        background = cv2.bitwise_or(background, padded)
        background, applied_article_boxes = resize_to_max_size(background, target_size, np.array(np.array([bbox])))
        return background, applied_article_boxes, np.array([article_label])

    def get_random_augmentations(self, image, bbox, number_of_agumentations):
        random_horizontal_flips = np.random.binomial(1, 0.5, number_of_agumentations)
        random_vertical_flips = np.random.binomial(1, 0.5, number_of_agumentations)
        random_luminocity = np.random.uniform(self.min_luminocity, self.max_luminocity, number_of_agumentations)
        images = [image]
        boxes = [bbox]
        for i in range(number_of_agumentations):
            augmented, aug_bbox = image, bbox
            if random_horizontal_flips[i]:
                augmented, aug_bbox = flip_horizontal(image, aug_bbox)
            if random_vertical_flips[i]:
                augmented, aug_bbox = flip_vertical(augmented, aug_bbox)
            augmented = adjust_illumination(augmented, random_luminocity[i])
            images += [augmented]
            boxes += [aug_bbox]
        return images, boxes

    def generate_random_batch(self, batch_size=32, real_proportion=0.25, target_size=448, seed=None):
        if seed is not None:
            np.random.seed(seed)
        number_augmentations = int(batch_size / (batch_size * real_proportion)) - 1
        all_images = []
        all_objects = []
        for i in range(int(batch_size * real_proportion)):
            image, bbox, labels = self.generate_random_image(target_size=target_size)
            aug_images, aug_boxes = self.get_random_augmentations(image, bbox, number_augmentations)
            all_labels = [labels for x in range(len(aug_images))]
            all_images += aug_images
            for i in range(len(aug_images)):
                objects = []
                for obj in range(len(all_labels[i])):
                    objects += [{
                        'name': all_labels[i][obj],
                        'ymin': np.min(aug_boxes[i][obj], axis=0)[0],
                        'xmin': np.min(aug_boxes[i][obj], axis=0)[1],
                        'ymax': np.max(aug_boxes[i][obj], axis=0)[0],
                        'xmax': np.max(aug_boxes[i][obj], axis=0)[1]
                    }]
                all_objects += [objects]
        return all_images, all_objects

if __name__ == '__main__':
    generator = ImageGenerator(images_location,
                               background_images_location,
                               object_proportion_range=[0.3, 0.9],
                               max_perspective_jitter=0.1,
                               objects_per_image_range=[3, 6],
                               max_object_overlap_area=0.2)
    #generator.get_random_article_photo('beer')
    #generator.get_random_background()
    #np.random.seed(42)
    batch_images, batch_bboxes, batch_labels = generator.get_random_batch(batch_size=4, seed=15)
    #image, boxes, labels = generator.generate_random_image()
    #flip_horizontal(image, boxes)
    #for i in range(100):
    #    plt.imsave(f'{i}.png', cv2.cvtColor(generator.generate_random_image()[0], cv2.COLOR_BGR2RGB))
    for i in range(len(batch_images)):
        plt.imsave("15_batch_" + i + ".png", cv2.polylines(batch_images[i], np.flip(batch_bboxes[i], axis=2), 1, (0, 255, 0), 3))
        print(batch_labels[i])

    batch_images, batch_bboxes, batch_labels = generator.get_random_batch(batch_size=4, seed=16)
    #image, boxes, labels = generator.generate_random_image()
    #flip_horizontal(image, boxes)
    #for i in range(100):
    #    plt.imsave(f'{i}.png', cv2.cvtColor(generator.generate_random_image()[0], cv2.COLOR_BGR2RGB))
    for i in range(len(batch_images)):
        plt.imsave("16_batch_" + i + ".png", cv2.polylines(batch_images[i], np.flip(batch_bboxes[i], axis=2), 1, (0, 255, 0), 3))
        print(batch_labels[i])
