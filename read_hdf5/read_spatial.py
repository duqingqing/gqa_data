import h5py
import numpy as np

if __name__ == '__main__':
    f = h5py.File('/home/dqq/下载/gqa_spatial.h5', 'r')
    dset = f['features']
    print(dset[0])

