import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# import sys
# sys.path.append('..')
import tensorflow as tf
from PIL import Image
import numpy as np
from get_feature_from_scene_graph.scene_to_tfrecord import FeatureToTFrecords


if __name__ == '__main__':
    tool = FeatureToTFrecords()
    tf_path = '../1170.tfrecords'
    # files = os.listdir(tf_path)
    # file_count = len([name for name in os.listdir(tf_path) if os.path.isfile(os.path.join(tf_path, name))])
    # for i in range(file_count):
    #     fileName= files[i]
    #     filePath = tf_path+fileName
    #     print(filePath)
    #     print("--------------------------------")
    data, lable = tool.load_tfrecords(srcfile=tf_path)
    print(np.shape(data))
    print((lable[0]))
    print("-----------------------------------")

