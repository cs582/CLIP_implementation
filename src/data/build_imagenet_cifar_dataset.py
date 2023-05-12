import pandas as pd
import numpy as np
import cv2
import json
from tqdm import tqdm
import os
import pickle
from PIL import Image

import matplotlib.pyplot as plt

def unpickle(file, dataset):
    if dataset == "cifar10":
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
    else:
        with open(file, 'rb') as fo:
            dict = pickle.load(fo)
    return dict

def load_dataset(data_folder, filename, dataset, img_size=64):
    data_file = os.path.join(data_folder, filename)

    data_col = 'data' if dataset=='imagenet' else b'data'
    label_col = 'labels' if dataset=='imagenet' else b'labels'

    d = unpickle(data_file, dataset)
    x = d[data_col]
    y = d[label_col]

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:,2*img_size2:]))
    x = x.reshape((len(x), img_size, img_size, 3))

    return x, y


def read_labels(iclabels):

    iclabels_map = { idx: "A low quality photo of a {}" for idx in range(1, 1001) }

    with open(iclabels, 'r') as f:
        for line in f:
            _, idx, label = line.split()
            idx = int(idx)
            iclabels_map[idx] = iclabels_map[idx].format( " ".join(label.split('_')) )

    return iclabels_map

def create_pairs(x, y, zero_index, imdir, queries):
    """
    Save images in an image directory and create csv file.
    :param x: imagenet images
    :param y: imagenet labels
    :param imdir: image directory
    :return: None
    """

    # Prompt engineering
    if not os.path.exists(imdir):
        os.mkdir(imdir)

    pairs = []
    for idx in tqdm(range(len(x)), total=len(x), desc="Saving images"):

        # Create image filename
        img_name = f'{idx+zero_index}.png'

        filename = os.path.join(imdir, img_name)

        # Pillow image save
        Image.fromarray(x[idx]).save(filename)

        # Append pairs
        pairs.append([queries[y[idx]], img_name])

    return pairs



def build(dataset):

    if dataset == "imagenet":
        datadir = f"data/imagenet/"
        queries = read_labels("data/imagenet/map_clsloc.txt")

    elif dataset == "cifar10":
        datadir = f"data/cifar10/"
        queries = {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck"
        }

    csv_filename = f"data/{dataset}/{dataset}.csv"

    imdir = os.path.join(datadir, "images")

    if dataset == "cifar10":
        pairs = []
        for batch in range(1, 6):
            x_temp, y_temp = load_dataset(datadir, f"data_batch_{batch}", img_size=32, dataset=dataset)
            pairs_temp = create_pairs(x_temp, y_temp, 10000*(batch-1), imdir, queries)
            pairs += pairs_temp
    else:
        x, y = load_dataset(datadir, 'val_data', img_size=64, dataset=dataset)
        pairs = create_pairs(x, y, 0, imdir, queries)

    df = pd.DataFrame(pairs, columns=["query", "image"])
    df.to_csv(csv_filename, index=False)
    print(f"Saved as {csv_filename}")

    return