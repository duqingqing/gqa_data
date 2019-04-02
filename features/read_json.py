import torchvision.transforms as transforms
import torch
import torch.cuda
import numpy as np
import json
from PIL import Image
if __name__ == '__main__':
    json_save_dir = '/media/zutnlp/49741bd8-e3dd-4326-bf71-0394aa198e95/zutnlpcv/gqa_tfrecords/feature_json/2386621.json'

    with open(json_save_dir, 'r') as load_f:
        load_dict = json.loads(load_f.read())
        feature_str = load_dict['2386621']['all_feature']
        nu_feature = np.array(feature_str)
        print(type(nu_feature))
        print(nu_feature.size())

        # feature = np.array2string(feature_str)
        # tensor_feature = torch.Tensor(feature)

        # print("---------------------------------------")
        # print(tensor_feature)
        # print(type(tensor_feature))
