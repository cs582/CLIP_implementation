import os
import cv2

import pandas as pd
import torch.utils.data

from tokenizers import Tokenizer
from urllib.request import urlopen
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def resize_image(image, h, w):
    resized_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized_image

# TODO: Implement the Dataset by using the csv file using in build_dataset.py
class ImageQueryDataset(Dataset):
    def __init__(self, dataset_file, tokenizer_file, img_resolution=(224, 224)):

        # Initial parameters
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(img_resolution),
        ])

        self.pairs = pd.read_csv(dataset_file, index_col=0).values
        self.tokenizer = Tokenizer.from_file(tokenizer_file)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        # Apply any data transformations if specified
        if self.transform is not None:
            image = self.transform(image)

        # Tokenize query
        token = torch.tensor(self.tokenizer.tokenize(query))

        return image, token
