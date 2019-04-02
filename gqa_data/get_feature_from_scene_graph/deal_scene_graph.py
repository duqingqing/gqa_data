import json

import numpy as np
from PIL import Image

from get_feature_from_scene_graph.getFeature import *

'''
把gqa数据中的scene_graph.json中对应图片的特征提取出来存为json 文件
'''
class GenerateFeatureFile(object):

    def __int__(self):
        pass


    def set_path(self,read_json_dir,image_dir, tfrecord_dir,json_save_dir):
        self.read_json_path = read_json_dir
        self.IMAGE_PATH = image_dir
        self.TFRECORD_PATH = tfrecord_dir
        self.Save_JSON_PAth = json_save_dir

    def set_picture_id(self,picture_id):
        self.picture_id = picture_id

    #打开图片
    def get_image(self,image_path):
        img = Image.open(image_path)
        return img
    #切图
    def cut_image(self,img,x,y,h,w):
        region = (x, y, h, w)
        cropImg = img.crop(region)
        return cropImg

    def read_scene_graph(self):
        model = make_model()
        with open(self.read_json_path,'r') as load_f:
            load_dict = json.loads(load_f.read())
            for item in load_dict:
                self.set_picture_id(item)
                #通过item 获取image的地址
                image_path = self.IMAGE_PATH+item+".jpg"
                #打开图片
                img = self.get_image(image_path)

                all_features = extract_feature(model,img)

                # 把numpy转为str
                all_features_str = (np.array(all_features).tolist())
                #把整张图片的特征放到item里
                load_dict[item]['all_feature'] = all_features_str

                objects = load_dict[item]["objects"]
                for object in objects:
                    x = objects[object]['x']
                    y = objects[object]['y']
                    h = objects[object]['h']
                    w = objects[object]['w']
                    cropImg = self.cut_image(img,x,y,h,w)
                    bbox_feature=extract_feature(model,cropImg)
                    #把numpy转为str
                    bbox_feature_str=np.array_str(bbox_feature)
                    #把bbox的特征放到object
                    objects[object]['bbox_feature']=bbox_feature_str
                    single_scene = {item:load_dict[item]}
                with open(self.Save_JSON_PAth+"%s.json"%self.picture_id, 'w') as f:
                    json.dump(single_scene, f)
                    print("deal number %s picture"%self.picture_id)
                    print("----------------------------------------")

if __name__ == '__main__':
    read_json_dir='/media/zutnlp/49741bd8-e3dd-4326-bf71-0394aa198e95/experiment/mac-network/data/sceneGraphs/train_sceneGraphs.json'
    image_dir='/media/zutnlp/49741bd8-e3dd-4326-bf71-0394aa198e95/experiment/mac-network/data/images/'
    tfrecord_dir=''
    json_save_dir='/media/zutnlp/49741bd8-e3dd-4326-bf71-0394aa198e95/zutnlpcv/gqa_tfrecords/feature_json/'
    generateJson = GenerateFeatureFile()
    generateJson.set_path(read_json_dir,image_dir, tfrecord_dir,json_save_dir)
    generateJson.read_scene_graph()
