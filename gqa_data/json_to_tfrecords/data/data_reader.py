
import tensorflow as tf
from tensorflow.python.data import Iterator
from data.data_utils import VisualGenomeDataUtils
from utils import dataset_util


class GQASceneGraphDataBuilder(object):
        def __init__(self):
                pass

        def read_and_decode(self,filename):
            #根据文件名生成一个队列
            filename_queue = tf.train.string_input_producer([filename])
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
            features = tf.parse_single_example(serialized_example,
                                               features={
                                                       # image data
                                                       'image/image_id': tf.FixedLenFeature([], tf.int64),
                                                       'image/height': tf.FixedLenFeature([], tf.int64),
                                                       'image/width': tf.FixedLenFeature([], tf.int64),
                                                       'image/feature': tf.FixedLenFeature([], tf.string),
                                                       # object
                                                       'image/object_list': tf.FixedLenFeature([], tf.string),
                                               })
            image_id = tf.cast(features['image/image_id'], tf.int32)
            image_height = tf.cast(features['image/height'],tf.int32)
            image_wight = tf.cast(features['image/width'],tf.int32)
            image_feature = tf.cast(features['image/feature'],tf.string)
            image_object = tf.cast(features['image/object_list'],tf.string)

            return image_id,image_height,image_wight,image_feature,image_object


if __name__ == '__main__':
    data_reader = GQASceneGraphDataBuilder()
    image_id, image_height, image_wight, image_feature, image_object = data_reader.read_and_decode('/home/dqq/下载/gqa_data/scene_graph_train.tfrecords')
    image_id, image_height, image_wight, image_feature, image_object = tf.train.shuffle_batch([image_id,image_height,image_wight,image_feature,image_object],
                                                    batch_size=30, capacity=2000,
                                                    min_after_dequeue=1000)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
            sess.run(init)
            threads = tf.train.start_queue_runners(sess=sess)
            for i in range(5):
                    id, height,wight,feature,object = sess.run([image_id, image_height, image_wight, image_feature, image_object ])
                    print(id,height,wight,feature,object)