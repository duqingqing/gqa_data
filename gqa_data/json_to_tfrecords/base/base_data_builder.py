# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import logging
from abc import ABCMeta, abstractmethod

import tensorflow as tf


class BaseDataBuilder(object):
    """
    Base Class for Data Prepare

    fetch raw data and convert them into TFRecord format

    """
    __metaclass__ = ABCMeta

    def __init__(self, data_config):
        self.data_config = data_config
        self._get_logger()

        # test_file = os.path.join(self.data_config.test_data_dir, 'test.tfrecords')
        # validation_file = os.path.join(self.data_config.valid_data_dir, 'validation.tfrecords')

        # self.test_writer = tf.python_io.TFRecordWriter(test_file)
        # self.valid_writer = tf.python_io.TFRecordWriter(validation_file)

    def _get_logger(self):
        logger = logging.getLogger("logger")
        logger.setLevel(logging.INFO)
        logging.basicConfig(format="%(message)s", level=logging.DEBUG)
        self.logger = logger

    def _int64_feature(self, value):
        """Wrapper for inserting an int64 Feature into a SequenceExample proto,
        e.g, An integer label.
        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        """Wrapper for inserting a bytes Feature into a SequenceExample proto,
        e.g, an image in byte
        """
        # return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature_list(self, values):
        """Wrapper for inserting an int64 FeatureList into a SequenceExample proto,
        e.g, sentence in list of ints
        """
        return tf.train.FeatureList(feature=[self._int64_feature(v) for v in values])

    def _bytes_feature_list(self, values):
        """Wrapper for inserting a bytes FeatureList into a SequenceExample proto,
        e.g, sentence in list of bytes
        """
        return tf.train.FeatureList(feature=[self._bytes_feature(v) for v in values])

    def _floats_feature(self, value):
        """Wrapper for inserting a float FeatureList into a Example proto,
            e.g,  in list of bytes
        """
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))


    def _write_tf_examples(self, tf_writer, tf_batch):
        for tf_example in tf_batch:
            tf_writer.write(tf_example.SerializeToString())
        print("saved {} tf_records".format(len(tf_batch)))
        pass

    def build_all_data(self):
        self.build_train_data()
        self.build_valid_data()
        self.build_test_data()
        pass

    @abstractmethod
    def _to_tf_example(self, mode, single_data):
        raise NotImplementedError()
        pass

    @abstractmethod
    def build_train_data(self):
        raise NotImplementedError()
        pass

    @abstractmethod
    def build_test_data(self):
        raise NotImplementedError()
        pass

    @abstractmethod
    def build_valid_data(self):
        raise NotImplementedError()
        pass
