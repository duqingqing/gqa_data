
import numpy as np
import json
from PIL import Image
if __name__ == '__main__':

    with open("/home/dqq/下载/sceneGraphs/val_sceneGraphs.json", 'r') as load_f:
        load_dict = json.loads(load_f.read())
        i=0
        for item in load_dict:
            i=i+1
        print(i)