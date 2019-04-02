# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from visgen.data.data_loader import VisualGenomeDataLoader
from visgen.data.data_utils import VisualGenomeDataUtils
from visgen.models import Region, Object


class DataDisplay(object):
    def __init__(self):
        self.data_loader = VisualGenomeDataLoader()
        self.data_utils = VisualGenomeDataUtils()
        self.fig = plt.gcf()
        self.fig.set_size_inches(18.5, 10.5)

    def _display(self, image_id, sub_images=None):
        """
        display raw image and its sub images
        :param image_raw:
        :param sub_images:
        :return:
        """
        image_raw = self.data_utils.load_image_raw(image_id=image_id)
        plt.imshow(image_raw)
        ax = plt.gca()
        for sub in sub_images:
            ax.add_patch(Rectangle((sub.x, sub.y),
                                   sub.width,
                                   sub.height,
                                   fill=False,
                                   edgecolor='red',
                                   linewidth=1))
            if isinstance(sub, Region):
                sub_txt = sub.phrase
            elif isinstance(sub, Object):
                sub_txt = sub.names

            ax.text(sub.x, sub.y, sub_txt, style='italic',
                    bbox={'facecolor': 'white',  'pad': 0})
        self.fig = plt.gcf()
        plt.tick_params(labelbottom='off', labelleft='off')
        plt.show()

    def display_images(self):
        data_gen = self.data_loader.load_images()
        for batch, data_batch in enumerate(data_gen):
            for image_data in data_batch:
                image_raw = self.data_utils.load_image_raw(image_id=image_data.image_id)
                self._display(image_raw=image_raw)

        pass

    # display regions in a image
    def display_image_regions(self, image_id):
        sub_images = self.data_loader.load_image_regions(image_id=image_id)
        self._display(image_id=image_id, sub_images=sub_images)
        pass

    # display objects in a image
    def display_image_objects(self, image_id):
        sub_images = self.data_loader.load_image_objects(image_id=image_id)
        self._display(image_id=image_id, sub_images=sub_images)
        pass


if __name__ == '__main__':
    display = DataDisplay()
    display.display_image_objects(image_id=3)
