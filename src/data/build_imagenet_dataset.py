import pandas as pd
import numpy as np
import cv2
import json
from tqdm import tqdm
import os
import pickle
from PIL import Image

import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def load_imagenet(data_folder, img_size=64):
    data_file = os.path.join(data_folder, 'val_data')

    d = unpickle(data_file)
    x = d['data']
    y = d['labels']

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:,2*img_size2:]))
    x = x.reshape((len(x), img_size, img_size, 3))

    return x, y


def read_labels(iclabels):

    iclabels_map = { idx: "A low quality photo of a {}" for idx in range(1, 1001) }

    with open(iclabels, 'r') as f:
        for line in f:
            _, idx, label = line.split()
            iclabels_map[idx].format( " ".join(label.split('_')) )

    return iclabels_map

def save_imagenet_locally(x, y, imdir, iclabels):
    """
    Save images in an image directory and create csv file.
    :param x: imagenet images
    :param y: imagenet labels
    :param imdir: image directory
    :return: None
    """

    # Prompt engineering
    queries = read_labels(iclabels)

    if not os.path.exists(imdir):
        os.mkdir(imdir)

    pairs = []
    for idx in tqdm(range(len(x)), total=len(x), desc="Saving ImageNet64 images"):

        # Create image filename
        img_name = f'{idx}.png'
        filename = os.path.join(imdir, img_name)

        # Pillow image save
        Image.fromarray(x[idx]).save(filename)

        # Append pairs
        pairs.append([queries[y[idx]], img_name])

    df = pd.DataFrame(pairs, columns=["query", "img"])
    df.to_csv("data/imagenet/imagenet.csv", index=False)


def build():

    datadir = "data/imagenet/"
    iclabels = "data/imagenet/map_clsloc.txt"

    imdir = os.path.join(datadir, "images")

    x, y = load_imagenet(datadir, img_size=64)
    save_imagenet_locally(x, y, imdir, iclabels)

    return