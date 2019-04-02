# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os

import tensorflow as tf

from visgen.base.base_data_builder import BaseDataBuilder
from visgen.data.data_config import VisualGenomeDataConfig
from visgen.data.data_loader import VisualGenomeDataLoader
from visgen.data.data_utils import VisualGenomeDataUtils
from visgen.feature.feature import FeatureManager

data_config = VisualGenomeDataConfig()


class VisualGenomeDataBuilder(BaseDataBuilder):

    def __init__(self, data_config):
        super(VisualGenomeDataBuilder, self).__init__(
            data_config=data_config
        )
        self.data_loader = VisualGenomeDataLoader()
        self.feature_manager = FeatureManager()
        pass

    def build_image_feature(self):
        """
            convert image data and visual feature into tfrecord format
        :return:
        """
        data_gen = self.data_loader.load_images()
        image_writer = tf.python_io.TFRecordWriter(self.data_config.image_feature_tf_file)
        for batch, batch_data in enumerate(data_gen):
            id_batch, shape_batch, image_raw_batch = batch_data
            features = self.feature_manager.get_vgg_feature(image_batch=image_raw_batch)
            for idx, id in enumerate(id_batch):
                vgg_feature = features[idx]
                tf_example = self._convert_image_to_tfexample(image_id=id, feature=vgg_feature)
                image_writer.write(tf_example.SerializeToString())  # Serialize To String
        image_writer.close()

    def _convert_image_to_tfexample(self, image_id, feature):
        features = tf.train.Features(feature={  # Non-serial data uses Feature
            "image/image_id": self._int64_feature(image_id),
            "image/vgg_feature": self._floats_feature(feature)
        })
        tf_example = tf.train.Example(features=features)
        return tf_example

    def build_region_feature(self):
        """
        Generate object visual feature for visual genome data set
        All object feature data are saved into tfrecord format
        :return:
        """
        data_gen = self.data_loader.load_regions()
        for batch, batch_data in enumerate(data_gen):
            for data in batch_data:
                image_id = data.image_id
                regions = data.regions
                print(regions)

                print(image_id)

    def generate_object_features(self):
        """
        Generate object visual feature for visual genome data set
        All object feature data are saved into tfrecord format
        :return:
        """


class VisualGenomeRegionGraphDataBuilder(BaseDataBuilder):
    """
    data builder for the scene graph of Visual Genome
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

    # data_prepare = VisualGenomeDataBuilder(data_config)
    # data_prepare.build_image_feature()

    region_data_prepare = VisualGenomeRegionGraphDataBuilder(data_config)
    region_data_prepare._build_data()



