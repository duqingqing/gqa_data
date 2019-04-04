# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os
import time
import ijson
import tensorflow as tf

from data.data_config import VisualGenomeDataConfig
from data.data_utils import VisualGenomeDataUtils


TOKEN_BEGIN = '<S>'
TOKEN_END = '</S>'


class VisualGenomeDataLoader(object):
    """
    Visual Genome Raw Dataset Loader
    """
    def __init__(self):
        self.config = VisualGenomeDataConfig()
        self.data_utils = VisualGenomeDataUtils()


    def load_scenegraph(self,filename):
        file_path = os.path.join(self.config.scene_graphs_json,filename)
        image_id = filename.split('.')[0]
        data = ()
        try:
            with open(file_path,mode='rb') as f:
                image_data_gen = ijson.items(f, "%s"%image_id)
                for image_gen in image_data_gen:
                    image_data = image_gen['all_feature']
                    image_object = image_gen['objects']
                    image_width = int(image_gen['width'])
                    image_height = int(image_gen['height'])
                    data = (image_id, image_data,image_object,image_width,image_height)
        except ijson.common.IncompleteJSONError as error:
            print('except: ijson.common.IncompleteJSONError')
        return data


    def batch_load_scenegraph(self):
        batch_contain=[]
        files = os.listdir(self.config.scene_graphs_json)
        for filename in files:
            batch_data = self.load_scenegraph(filename)
            if len(batch_data)>0 :
                batch_contain.append(batch_data)
            else:
                continue
            if len(batch_contain) == self.config.batch_size:
                yield batch_contain
                batch_contain = []
        if len(batch_contain) > 0:
            yield batch_contain
        del batch_contain



    # def load_regions(self):
    #     batch_data = []
    #     with open(file=self.config.region_descriptions_json, mode='rb') as f:
    #         data_gen = ijson.items(f, "item")
    #         for data in data_gen:
    #             image_id = data['id']
    #             regions = data['regions']
    #             for r in regions:
    #                 batch_data.append(r)
    #                 if len(batch_data) == self.config.batch_size:
    #                     yield batch_data
    #                     batch_data = []
    #         if len(batch_data) > 0:
    #             yield batch_data
    #         del batch_data

if __name__ == '__main__':

    data_loader = VisualGenomeDataLoader()
    scene_data_gen = data_loader.batch_load_scenegraph()
    for batch, batch_datas in enumerate(scene_data_gen):
         print(len(batch_datas[0]))
        # if len(batch_datas[0])>0:
        #     for batch_index, (image_id, image_data, image_object,image_width,image_height) in enumerate(batch_datas):
        #         print(image_id)
        #         print(image_width)
        #         print(image_height)
        #         print('-------------------------------------------')
        # else:
        #     continue






