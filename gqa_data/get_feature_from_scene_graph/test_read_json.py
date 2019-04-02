import numpy as np
import json
from get_feature_from_scene_graph.scene_to_tfrecord import FeatureToTFrecords
import json

import numpy as np

from get_feature_from_scene_graph.scene_to_tfrecord import FeatureToTFrecords

'''
测试读取json里的特征信息
'''

class ParsingSceneData(object):
    def __int__(self):
        pass

    #将list转为numpy
    def list_to_tensor(self,list_data):
        np_data = np.array(list_data)
        return np_data


if __name__ == '__main__':
    tool = FeatureToTFrecords()
    psd = ParsingSceneData()
    json_save_dir = '/home/dqq/下载/gqa_data/feature_json/1170.json'
    with open(json_save_dir, 'r') as load_f:
        load_dict = json.loads(load_f.read())
        all_feature= load_dict['1170']['all_feature']
        ten_f = psd.list_to_tensor(all_feature)
        tool.save_tfrecords(ten_f,[str.encode('1170')],'../1170.tfrecords')






