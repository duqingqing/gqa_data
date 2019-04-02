# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os

import tensorflow as tf

from base.base_data_builder import BaseDataBuilder
from data.data_config import VisualGenomeDataConfig
from data.data_loader import VisualGenomeDataLoader
from data.data_utils import VisualGenomeDataUtils

data_config = VisualGenomeDataConfig()


class VisualGenomeRegionGraphDataBuilder(BaseDataBuilder):
    """
    data builder for the scene graph of GQA
    """

    def __init__(self, data_config):
        super(VisualGenomeRegionGraphDataBuilder, self).__init__(
            data_config=data_config
        )
        self.data_loader = VisualGenomeDataLoader()
        self.data_utils = VisualGenomeDataUtils()

    def __write_tf_examples(self, data_writer, examples):
        for idx, example in enumerate(examples):
            data_writer.write(example)

    def _build_data(self):
        region_graph_gen = self.data_loader.load_region_graphs()

        data_file = os.path.join(self.data_config.train_data_dir, 'train.tfrecords')
        data_writer = tf.python_io.TFRecordWriter(data_file)

        tf_batch = list()
        batch_size = 100
        for batch, batch_data in enumerate(region_graph_gen):
            for image_idx, (image_id, region_graphs) in enumerate(batch_data):
                image_raw_data = self.data_utils.load_image(image_id)
                for region_idx, region_graph in enumerate(region_graphs):
                    data = (image_id, image_raw_data, region_graph)
                    tf_example = self._to_tf_example(data)
                    tf_batch.append(tf_example)
                    if len(tf_batch) == batch_size:
                        self.__write_tf_examples(data_writer=data_writer,examples=tf_batch)
                        tf_batch = list()
        if len(tf_batch) > 0:
            self.__write_tf_examples(data_writer=data_writer,examples=tf_batch)
        del tf_batch
        pass

    def _to_tf_example(self, data):
        """
        convert data to tfrecord example
        :param data: region_graph json data
        :return:
        """
        image_id, image_raw_data, region_graph = data
        region_id = region_graph["region_id"]
        objects = region_graph["objects"]
        relationships = region_graph["relationships"]

        pass









if __name__ == '__main__':
    data_config = VisualGenomeDataConfig()
    region_data_prepare = VisualGenomeRegionGraphDataBuilder(data_config)
    region_data_prepare._build_data()



