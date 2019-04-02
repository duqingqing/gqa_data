import os

import numpy as np
import skimage.color
import skimage.io

from visgen.data.data_config import VisualGenomeDataConfig


class VisualGenomeDataUtils(object):
    """
    Visual Genome DataUtils Class

    """

    def __init__(self):
        self.image_dict = dict()
        pass

    def load_image_raw(self, image_id):
        if image_id in self.image_dict:
            image_raw = self.image_dict[image_id]
        else:
            image_raw = self.load_image(image_id)
            self.image_dict[image_id] = image_raw
        return image_raw

    def load_image(self, image_id):

        """
        load image data base on given image id
        :param image_id: image id
        :return: image data
        """
        current_image = skimage.io.imread(os.path.join(VisualGenomeDataConfig.image_dir, str(image_id) + ".jpg"))
        if len(current_image.shape) == 2 or current_image.shape[2] == 1:  # this is to convert a gray to RGB image
            current_image = skimage.color.gray2rgb(current_image)  #
        return current_image


    def get_object_raw(self, object):
        pass

    def get_object_raw(self, image):
        """
        get raw data list of all objects in a given image
        :param object:
        :param image:
        :return:
        """
        pass

    def load_region(self, region, current_image):
        """
        get region from current image
        :param region: region data
        :param current_image: current image
        :return: region image data
        """
        region_image = current_image[region.y: region.y + region.height, region.x: region.x + region.width, :]
        # region_image = resize(region_image, (224, 224), mode='reflect')
        return region_image

    def image_crop(self, raw_image, sub_image_data):
        """
        :param raw_image:
        :param x:
        :param y:
        :param width:
        :param height:
        :return:
            cropped_image
        """
        # x, y are top left coordinates
        xmin = int(sub_image_data['x'])
        ymax = int(sub_image_data['y'])

        if 'width' in sub_image_data.keys():
            xmax = xmin + int(sub_image_data['width'])
        else:
            xmax = xmin + int(sub_image_data['w'])

        if 'height' in sub_image_data.keys():
            ymin = ymax - int(sub_image_data['height'])
        else:
            ymin = ymax - int(sub_image_data['h'])

        crop_im = np.zeros(raw_image.shape)
        crop_im[xmin:xmax, ymin:ymax, :] = raw_image[xmin:xmax, ymin:ymax, :]
        cropped_shape = crop_im.shape
        if cropped_shape[0] <= 0 or cropped_shape[1] <= 0:
            print("raw_shape={}, cropped_shape={}".format(raw_image.shape, cropped_shape))
            crop_im = None
        return crop_im
