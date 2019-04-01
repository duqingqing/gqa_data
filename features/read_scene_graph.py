from getFeature import *
import numpy as np
import json
from PIL import Image

class GenerateFeatureFile(object):

    def __int__(self,read_json_dir,image_dir, tfrecord_dir,json_save_dir):
        self.read_json_path = read_json_dir
        self.IMAGE_PATH = image_dir
        self.TFRECORD_PATH = tfrecord_dir
        self.Save_JSON_PAth = json_save_dir

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
                all_features_str =np.array_str(all_features)
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
        with open(self.Save_JSON_PAth+"%s.json"%self.picture_id, 'w') as f:
                json.dump(load_dict, f)

if __name__ == '__main__':
    read_json_dir=''
    image_dir=''
    tfrecord_dir=''
    json_save_dir=''
    generateJson = GenerateFeatureFile()
    generateJson.set_path(read_json_dir,image_dir, tfrecord_dir,json_save_dir)
    generateJson.read_scene_graph()
