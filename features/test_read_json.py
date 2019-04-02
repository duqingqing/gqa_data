import torchvision.transforms as transforms
import torch
import torch.cuda
import numpy as np
import json
from PIL import Image

if __name__ == '__main__':
    json_save_dir = '/home/dqq/下载/gqa_data/1170.json'
    with open(json_save_dir, 'r') as load_f:
        load_dict = json.loads(load_f.read())
        all_feature= load_dict['2407890']['all_feature']
        print("before change....")
        print(all_feature)
        print(type(all_feature))
        print('-----------------------------------------')
        np_f = np.array(all_feature)
        print("after 1 change....")
        print(np_f)
        print(type(np_f))
        print(np.shape(np_f))
        print('-----------------------------------------')

        ten_f = torch.Tensor(np_f)
        print("after 2 change....")
        print(ten_f)
        print(type(ten_f))
        print('-----------------------------------------')





