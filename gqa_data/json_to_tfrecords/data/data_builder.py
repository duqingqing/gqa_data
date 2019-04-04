# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding
from utils import dataset_util
import os
import numpy as np
import tensorflow as tf

from base.base_data_builder import BaseDataBuilder
from data.data_config import VisualGenomeDataConfig
from data.data_loader import VisualGenomeDataLoader
from data.data_utils import VisualGenomeDataUtils

data_config = VisualGenomeDataConfig()


class GQASceneGraphDataBuilder(BaseDataBuilder):
    """
    data builder for the scene graph of GQA
    """

    def __init__(self, data_config):
        super(GQASceneGraphDataBuilder, self).__init__(
            data_config=data_config
        )
        self.data_loader = VisualGenomeDataLoader()
        self.data_utils = VisualGenomeDataUtils()

    def __write_tf_examples(self, data_writer, tf_batch):
        for idx, example in enumerate(tf_batch):
            data_writer.write(example)
        print("saved {} tf_records".format(len(tf_batch)))
        pass

    def _build_data(self):
        scene_data_gen = self.data_loader.batch_load_scenegraph()
        data_file = os.path.join(self.data_config.train_data_dir, 'scene_graph_train.tfrecords')
        data_writer = tf.python_io.TFRecordWriter(data_file)
        tf_batch = list()
        batch_size = 100
        for batch, batch_datas in enumerate(scene_data_gen):
            for index, (image_id, image_data,image_object,image_width,image_height) in enumerate(batch_datas):
                print('make number {} image`s scene-graph to tfrecords'.format(image_id))
                data = (image_id, image_data, image_object,image_width,image_height)
                tf_example = self._to_tf_example(data)
                tf_batch.append(tf_example)
                if len(tf_batch) == batch_size:
                    self.__write_tf_examples(data_writer, tf_batch)
                    tf_batch = list()
        if len(tf_batch) > 0:
            self.__write_tf_examples(data_writer, tf_batch)
        del tf_batch
        pass


    def _to_tf_example(self, data):
        """
        convert data to tfrecord example
        :param data: region_graph json data
        :return:
        """
        image_id, image_data, image_object,image_width,image_height = data
        image_data = np.asarray(image_data)
        object_list = list()
        for object in image_object:
            object_id = object
            object_name = image_object[object]["name"]
            object_h = image_object[object]["h"]
            object_w = image_object[object]["w"]
            object_x = image_object[object]["x"]
            object_y = image_object[object]["y"]
            object_attributes = image_object[object]["attributes"]
            object_relations = image_object[object]["relations"]
            object_feature = image_object[object]["bbox_feature"]
            single_object = [object_id,object_name,object_h,object_w,object_x,object_y,object_attributes,object_relations,object_feature]
            object_list.append(single_object)
        object_list_np_data = np.asarray(object_list)
        object_list_np_bytes = object_list_np_data.tobytes()

        features = tf.train.Features(feature={
            # image data
            'image/image_id': dataset_util.int64_feature(int(image_id)),
            'image/height': dataset_util.int64_feature(image_height),
            'image/width': dataset_util.int64_feature(image_width),
            'image/feature': dataset_util.bytes_feature(image_data.tobytes()),
            # object
            'image/object_list': dataset_util.bytes_feature(object_list_np_bytes),
        })
        tf_example = tf.train.Example(features=features)
        return  tf_example.SerializeToString()

if __name__ == '__main__':
    data_config = VisualGenomeDataConfig()
    region_data_prepare = GQASceneGraphDataBuilder(data_config)
    region_data_prepare._build_data()



