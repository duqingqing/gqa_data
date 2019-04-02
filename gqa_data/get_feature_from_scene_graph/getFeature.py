import os
import numpy as np
import sys
sys.path.append('..')
import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms
from PIL import Image
import json
from get_feature_from_scene_graph.scene_to_tfrecord import FeatureToTFrecords

'''
使用pytorch训练好的模型提取图片的特征
'''

img_to_tensor = transforms.ToTensor()


def make_model():
    resmodel = models.resnet101(pretrained=True)
    resmodel.cuda()  # 将模型从CPU发送到GPU,如果没有GPU则删除该行
    return resmodel


def is_color_image(im):
    pix=im.convert('RGB')
    width=im.size[0]
    height=im.size[1]
    oimage_color_type=False
    is_color=[]
    for x in range(width):
        for y in range(height):
            r,g,b=pix.getpixel((x,y))
            r=int(r)
            g=int(g)
            b=int(b)
            if (r==g) and (g==b):
                pass
            else:
                oimage_color_type=True
    return oimage_color_type

# 特征提取
def extract_feature(resmodel, img):
    resmodel.fc = torch.nn.LeakyReLU(0.1)
    resmodel.eval()
    color = is_color_image(img)
    if not color:
        img = img.convert('RGB')
    else:
        pass
    img = img.resize((224, 224))
    img_tensor = img_to_tensor(img)
    img_tensor = img_tensor.reshape(1, 3, 224, 224)

    img_tensor = img_tensor.cuda()
    result = resmodel(Variable(img_tensor))
    result_npy = result.data.cpu().numpy()

    return result_npy
#写文件
def store(data,json_file_dir):
    with open(json_file_dir, 'w') as json_file:
        json_file.write(json.dumps(data))

if __name__ == "__main__":
    image_file="/home/dqq/图片/test_image/";
    tf_path = '/home/dqq/图片/feature'
    files = os.listdir(image_file)
    file_count = len([name for name in os.listdir(image_file) if os.path.isfile(os.path.join(image_file, name))])
    for i in range(file_count):
        fileName,fileType = files[i].split('.')
        lable = str.encode(fileName)
        print(fileName)
        print("--------------------------------------")
        model = make_model()
        image_path = image_file+fileName+'.'+fileType
        feature = extract_feature(model,image_path)
        # tool = FeatureToTFrecords()
        # tool.save_tfrecords(feature,[lable],tf_path+"/%s.tfrecords"%(fileName))
    # data,lable =tool.load_tfrecords(srcfile="feature.tfrecords")
    # print(np.shape(data))
    # print(lable)




