import pandas as pd
import numpy as np
import json
import os
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def load_imagenet(data_folder, idx, img_size=64):
    data_file = os.path.join(data_folder, 'val_data')

    d = unpickle(data_file)
    x = d['data']
    y = d['labels']

    return x, y

def build():

    datadir = "data/imagenet/"

    image_net = load_imagenet(datadir, 0, img_size=64)

    return