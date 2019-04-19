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


    def set_path(self,read_json_dir,image_dir, question_json_dir,json_save_dir,new_json_save_dir):
        self.read_json_path = read_json_dir
        self.IMAGE_PATH = image_dir
        self.question_path =question_json_dir
        self.Save_JSON_PAth = json_save_dir
        self.new_json_save_path = new_json_save_dir

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

    #向制定image_id对应的json文件添加question
    def add_question_to_scenegraph(self,image_id,questions_data):
        #打开现有的scene_graph数据
        try:
            with open(self.Save_JSON_PAth+"%s.json"%image_id,'r') as load_f:
                load_dict = json.loads(load_f.read())
                for item in load_dict:
                    #添加一个问题字段
                    load_dict[item]['questions'] = questions_data
                    single_scene = {item: load_dict[item]}
                    #保存到新的文件中
                    with open(self.new_json_save_path+"%s.json"%image_id, 'w') as f:
                        json.dump(single_scene, f)
                        print("make number %s picture"%image_id)
                        print("----------------------------------------")
        except:
            print("%s.json is not exist"%image_id)
            pass


    def write_question_to_scenegraph(self):
        question_json_collects = []  # 用来装载同一个image的问题数据
        image_mark = 0 #标记当前的图片id
        stepcount = 0 #标记循环的开始
        with open(self.question_path,'r') as question_file:
            question_datas = json.loads(question_file.read())
            questions = question_datas['questions']
            for question_item in questions :
                imageId = question_item['imageId']
                if stepcount==0:
                    image_mark = imageId
                    question_json_collects.append(question_item)
                    stepcount=stepcount+1
                else:
                    if imageId==image_mark:
                        question_json_collects.append(question_item)
                    else:
                        #到此一个图片的所有问题在list中了，进行保存
                        self.add_question_to_scenegraph(image_mark,question_json_collects)
                        image_mark = imageId
                        question_json_collects = []
                        question_json_collects.append(question_item)


if __name__ == '__main__':
    read_json_dir='/media/zutnlp/49741bd8-e3dd-4326-bf71-0394aa198e95/experiment/mac-network/data/sceneGraphs/train_sceneGraphs.json'
    image_dir='/media/zutnlp/49741bd8-e3dd-4326-bf71-0394aa198e95/experiment/mac-network/data/images/'
    question_json_dir='/home/dqq/下载/gqa_data/all_val_data.json'
    json_save_dir='/media/zutnlp/49741bd8-e3dd-4326-bf71-0394aa198e95/zutnlpcv/gqa_tfrecords/feature_json/'
    new_json_save_dir = '/media/zutnlp/49741bd8-e3dd-4326-bf71-0394aa198e95/zutnlpcv/gqa_tfrecords/new_feature_json/'
    generateJson = GenerateFeatureFile()
    generateJson.set_path(read_json_dir,image_dir, question_json_dir,json_save_dir,new_json_save_dir)
    generateJson.write_question_to_scenegraph()
