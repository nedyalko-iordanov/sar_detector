import numpy as np
import cv2
import json
from matplotlib import pyplot as plt
import math
import os
import shutil

def crop_object(image, contour_points):
  """
  Method that will crop the object by the passed contour points by applying
  a black background and doing a min area crop rectangle and aligning it
  to be orthogonal to the frame.
  """
  contour_crop = crop_by_contour(image, contour_points)
  min_area_crop = crop_by_min_area_rect(contour_crop, contour_points)
  return min_area_crop

def crop_by_contour(image, contour_points):
  """
  Method to apply a black background to points outside of the pased contours
  """
  stencil = np.zeros(image.shape).astype(image.dtype)
  mask = [255, 255, 255]
  cv2.fillPoly(stencil, [contour_points], mask)
  result = cv2.bitwise_and(image, stencil)
  return result

def crop_by_min_area_rect(image, points):
  """
  Method to identify the min area bounding rectangle, crop the image by it
  and align the rectangle to be orthogonal to the frame.
  """
  # get the min area rect for the object
  rect = cv2.minAreaRect(points)
  box = cv2.boxPoints(rect)
  box = np.int0(box)
  # calculate it's height and width
  height = math.sqrt(math.pow(box[0, 0] - box[1, 0], 2) + math.pow(box[0, 1] - box[1, 1], 2))
  width = math.sqrt(math.pow(box[0, 0] - box[3, 0], 2) + math.pow(box[0, 1] - box[3, 1], 2))
  # define center for rotation
  center = box[1]
  # calculate cosine of the angle
  cosine = (box[3, 0] - box[0, 0]) / width
  # get the angle of rotation
  alpha = math.acos(cosine)
  alpha = math.degrees(alpha)
  # coordinate of the points in box points after the rectangle has been
  # straightened
  dst_pts = np.array([[0, height-1],
                      [0, 0],
                      [width-1, 0],
                      [width-1, height-1]], dtype="float32")
  # the perspective transformation matrix
  M = cv2.getPerspectiveTransform(box.astype("float32"), dst_pts)
  # directly warp the rotated rectangle to get the straightened rectangle
  warped = cv2.warpPerspective(image, M, (int(width), int(height)))
  return warped

def augment_and_save(image, image_name, storage_location, illumination_steps, rotation_steps):
  """
  Method to apply all combinations of illuminations and rotations passed to the given image
  and save the image with a specific name to the storage location.
  """
  for illumination in illumination_steps:
    for rotation in rotation_steps:
      augmented = adjust_illumination(image, illumination)
      augmented = rotate_image(augmented, rotation)
      path = os.path.join(storage_location, image_name + "_l-" + str(illumination) + "_r-" + str(rotation) + ".jpg")
      cv2.imwrite(path, augmented)

def adjust_illumination(image, gamma):
  """ Adjust the illumination of an image by gamma"""
  # Make lookup table for gamma
  inv_gamma = 1.0 / gamma
  table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)])
  # Match the image with lookup table
  adjusted = cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))
  return adjusted

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def resize_to_match_area(image, area):
    height, width, _ = image.shape
    a = height / width
    target_height = int(math.sqrt(area / a))
    target_width = int(a * target_height)
    resized = cv2.resize(image, (target_height, target_width))
    return resized

def resize_to_max_size(image, max_size, boxes=None):
    height, width, _ = image.shape
    if height > width:
        ratio = max_size / height
        target_height = max_size
        target_width = width/height * target_height
    else:
        ratio = max_size / width
        target_width = max_size
        target_height = height/width * target_width
    resized = cv2.resize(image, (int(target_width), int(target_height)))
    if boxes is not None:
        boxes[:, :, 0] = boxes[:, :, 0] * ratio
        boxes[:, :, 1] = boxes[:, :, 1] * ratio
        boxes = boxes.astype('int')
    return resized, boxes


def apply_random_perspective(image, max_perspective_jitter):
    height, width, _ = image.shape
    perspective_jitter_x = max_perspective_jitter * width
    perspective_jitter_y = max_perspective_jitter * height
    perspective_jitter_x_bl = np.random.uniform(-perspective_jitter_x, perspective_jitter_x)
    perspective_jitter_y_bl = np.random.uniform(-perspective_jitter_y, perspective_jitter_y)
    perspective_jitter_x_tl = np.random.uniform(-perspective_jitter_x, perspective_jitter_x)
    perspective_jitter_y_tl = np.random.uniform(-perspective_jitter_y, perspective_jitter_y)
    perspective_jitter_x_tr = np.random.uniform(-perspective_jitter_x, perspective_jitter_x)
    perspective_jitter_y_tr = np.random.uniform(-perspective_jitter_y, perspective_jitter_y)
    perspective_jitter_x_br = np.random.uniform(-perspective_jitter_x, perspective_jitter_x)
    perspective_jitter_y_br = np.random.uniform(-perspective_jitter_y, perspective_jitter_y)
    new_x_bl = 0 + perspective_jitter_x_bl
    new_y_bl = 0 + perspective_jitter_y_bl
    new_x_tl = 0 + perspective_jitter_x_tl
    new_y_tl = height + perspective_jitter_y_tl
    new_x_tr = width + perspective_jitter_x_tr
    new_y_tr = height + perspective_jitter_y_tr
    new_x_br = width + perspective_jitter_x_br
    new_y_br = 0 + perspective_jitter_y_br
    source_points = np.array(
        [[0, 0],
         [0, height],
         [width, height],
         [width, 0]]).astype(np.float32)
    dst_points = np.array(
        [[new_x_bl, new_y_bl],
         [new_x_tl, new_y_tl],
         [new_x_tr, new_y_tr],
         [new_x_br, new_y_br]]).astype(np.float32)
    diff = source_points - dst_points
    addition = np.max(diff, axis=0)
    dst_points = dst_points + addition
    dims = np.max(dst_points, axis=0).astype(np.int32)
    M = cv2.getPerspectiveTransform(np.array(source_points).astype(np.float32), np.array(dst_points).astype(np.float32))
    warped = cv2.warpPerspective(image, M, (dims[0], dims[1]))
    return warped


def crop_black_pixels(image):
    non_black_pixels = np.argwhere(np.max(image != np.array([0, 0, 0]), axis=2))
    bl_y, bl_x = np.min(non_black_pixels, axis=0)
    tr_y, tr_x = np.max(non_black_pixels, axis=0)
    image = image[bl_y:tr_y, bl_x:tr_x]
    return image

def calculate_iou(box1, box2):
    height1, width1 = np.max(box1, axis=0) - np.min(box1, axis=0)
    height2, width2 = np.max(box2, axis=0) - np.min(box2, axis=0)
    miny, minx = np.min(np.append(box1, box2, axis=0), axis=0)
    maxy, maxx = np.max(np.append(box1, box2, axis=0), axis=0)
    y_intersection = max((height1 + height2) - (maxy - miny), 0)
    x_intersection = max((width1 + width2) - (maxx - minx), 0)
    intersection = y_intersection * x_intersection
    union = height1 * width1 + height2 * width2 - intersection
    return intersection / union

def alphaBlend(img1, img2, mask):
    """ alphaBlend img1 and img 2 (of CV_8UC3) with mask (CV_8UC1 or CV_8UC3)
    """
    if mask.ndim==3 and mask.shape[-1] == 3:
        alpha = mask/255.0
    else:
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
    return blended

def flip_horizontal(image, bbox):
    image = np.flip(image, axis=1)
    image = np.ascontiguousarray(image, dtype=np.uint8)
    _, width, _ = image.shape
    sub = np.array([[[0, width]]])
    bbox = np.abs(sub - bbox)
    return image, bbox

def flip_vertical(image, bbox):
    image = np.flip(image, axis=0)
    image = np.ascontiguousarray(image, dtype=np.uint8)
    height, _, _ = image.shape
    sub = np.array([[[height, 0]]])
    bbox = np.abs(sub - bbox)
    return image, bbox


class ImageGenerator:
    def __init__(
            self,
            article_images_path,
            background_images_path,
            output_resolution_range=[352, 448],  # taken from https://github.com/experiencor/keras-yolo3 config file
            objects_per_image_range=[1, 5],
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
        self.min_objects, self.max_objects = objects_per_image_range
        self.min_object_proportion, self.max_object_proportion = object_proportion_range
        self.min_visible_object_area = min_visible_object_area
        self.max_overlap_area = max_object_overlap_area
        self.max_perspective_jitter = max_perspective_jitter
        self.min_luminocity, self.max_luminocity = luminocity_range
        self.article_images, self.labels = self.get_cropped_article_images()
        self.backgrounds = self.get_backgrounds()

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
        if self.max_objects > 1:
            number_of_objects = np.random.randint(self.min_objects, self.max_objects)
        else:
            number_of_objects = 1
        articles = np.random.choice(list(self.article_images.keys()), number_of_objects, replace=True)
        article_images = [self.get_random_article_photo(article) for article in articles]
        background = self.get_random_background()
        background_height, background_width, _ = background.shape
        applied_article_boxes = []
        applied_article_labels = []
        for i, article in enumerate(article_images):
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
            article_label = articles[i]
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
            max_iou = max(calculate_iou(bbox, x) for x in applied_article_boxes) if i != 0 else 0.0
            if max_iou >= self.max_overlap_area:
                #print(max_iou)
                pass
            else:
                applied_article_boxes += [bbox]
                applied_article_labels += [article_label]
                padded = np.pad(article, ((article_bl[0] + rest_height, background_height - 1 - article_tr[0]),
                                          (article_bl[1] + rest_width, background_width - 1 - article_tr[1]),
                                          (0, 0)))
                inv_mask = np.min(padded == np.array([0, 0, 0]), axis=2).astype(np.int8)
                background = cv2.bitwise_and(background, background,  mask=inv_mask)
                background = cv2.bitwise_or(background, padded)
        background, applied_article_boxes = resize_to_max_size(background, target_size, np.array(applied_article_boxes))
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        return background, np.array(applied_article_boxes), np.array(applied_article_labels)

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

    def generate_random_batch(self, batch_size=32, augmented_proportion=0.5, target_size=448, seed=None):
        if seed is not None:
            np.random.seed(seed)
        number_augmentations = int(batch_size / (batch_size * augmented_proportion)) - 1
        all_images = []
        all_objects = []
        for i in range(int(batch_size * augmented_proportion)):
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
    # define where are our images located
    images_location = 'C:\\Users\\Freeware Sys\\Desktop\\articles\\raw_debug'
    background_images_location = 'C:\\Users\\Freeware Sys\\Desktop\\articles\\backgrounds_debug'
    # define where we want the images for the database to be located
    database_location = 'C:\\Users\\Freeware Sys\\Desktop\\articles\\cropped'
    try:
        os.mkdir(database_location)
    except FileExistsError:
        print(database_location + ' directory already exists')
    # define steps for illumination and rotation angles
    illumination_steps = [1.0]
    # rotation_steps = [0, 45, 90, 135, 180, 225, 270, 315]
    rotation_steps = [0]
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
