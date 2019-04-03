# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os
from abc import ABCMeta, abstractmethod

import tensorflow as tf
from tensorflow.python.data import Iterator

class BaseDataReader(object):
    __metaclass__ = ABCMeta

    def __init__(self, data_config):
        self.data_config = data_config

        # default batch_size from data reader config
        self._batch_size = self.data_config.reader_batch_size
        self._build_context_and_feature()
        self.training_dataset = self._get_dataset(self.data_config.train_tfrecord_dir)

        self.iterator = Iterator.from_structure(self.training_dataset.output_types,
                                                self.training_dataset.output_shapes)
        pass

    def get_next_batch(self, batch_size=None):
        if batch_size:
            self._batch_size = batch_size
        return self.iterator.get_next()

    def get_train_init_op(self):
        if not self.training_dataset:
            self.training_dataset = self._get_dataset(self.data_config.train_tfrecord_dir)
        return self.iterator.make_initializer(self.training_dataset)
        pass

    def get_infer_init_op(self):
        infer_dataset = self._get_dataset(self.data_config.test_tfrecord_dir)
        return self.iterator.make_initializer(infer_dataset)
        pass

    def get_valid_init_op(self):
        valid_dataset = self._get_dataset(self.data_config.valid_tfrecord_dir)
        return self.iterator.make_initializer(valid_dataset)
        pass

    def _get_dataset(self, data_dir):
        """
        get TFRecordDataset from give data_dir and mapping them into dataset
        :param data_dir:
        :return:
        """
        print("tf data_reader initialize tf_dataset from {}.".format(data_dir))
        filenames = os.listdir(data_dir)
        data_files = []
        for filename in filenames:
            data_file = os.path.join(data_dir, filename)
            print("\tinitializing dataset from {}.".format(filename))
            data_files.append(data_file)
        dataset = tf.data.TFRecordDataset(data_files)
        # parsing tf_record
        dataset = dataset.map(map_func=self._parse_tf_example)

        # mapping dataset
        dataset = self._mapping_dataset(dataset)  # mapping to target format
        return dataset

    @abstractmethod
    def _mapping_dataset(self, dataset):
        """mapping data to direable """
        raise NotImplementedError()
        pass

    @abstractmethod
    def _build_context_and_feature(self):
        """used by tfrecord parsing"""
        raise NotImplementedError()
        pass

    @abstractmethod
    def _parse_tf_example(self, tf_example):
        """parsing tfrecord parsing"""
        raise NotImplementedError()
        pass

    pass
