import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# import sys
# sys.path.append('..')
import tensorflow as tf
from PIL import Image
import numpy as np
import threading


class FeatureToTFrecords(object):
    temp_file_list = []
    step = 100
    index = 0

    def __int__(self):
        pass

    def setPath(self, tfrecordPath):
        #self.IMAGE_PATH = imagePath
        self.TFRECORD_PATH = tfrecordPath

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def save_tfrecords(self,data, label, desfile):
        with tf.python_io.TFRecordWriter(desfile) as writer:
            for i in range(len(data)):
                features = tf.train.Features(
                    feature={
                        "data": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[data[i].astype(np.float32).tostring()])),
                        "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label[i]]))
                    }
                )
                example = tf.train.Example(features=features)
                serialized = example.SerializeToString()
                writer.write(serialized)

    def _parse_function(self,example_proto):
        features = {"data": tf.FixedLenFeature((), tf.string),
                    "label": tf.FixedLenFeature((), tf.string)}
        parsed_features = tf.parse_single_example(example_proto, features)
        data = tf.decode_raw(parsed_features['data'], tf.float32)
        return data, parsed_features["label"]

    def load_tfrecords(self,srcfile):
        sess = tf.Session()
        dataset = tf.data.TFRecordDataset(srcfile)  # load tfrecord file
        dataset = dataset.map(self._parse_function)  # parse data into tensor
        dataset = dataset.repeat(2)  # repeat for 2 epoches
        dataset = dataset.batch(5)  # set batch_size = 5

        iterator = dataset.make_one_shot_iterator()
        next_data = iterator.get_next()

        while True:
            try:
                data, label = sess.run(next_data)
                return data ,label
            except tf.errors.OutOfRangeError:
                break
