import os

import numpy as np
import pandas as pd
import torch.utils.data

from tokenizers import Tokenizer
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.io import read_image
import torchvision.transforms as T


class ImageQueryDataset(Dataset):
    def __init__(self, dataset_file, image_path, tokenizer_file, img_res=112):
        self.img_res = img_res

        # Set image path
        self.image_path = image_path

        # Transformation is only Random Crop to match image resolution
        self.transform = transforms.Compose([
            transforms.RandomCrop((img_res, img_res)),
        ])

        # Load tokenizer from file
        self.tokenizer = Tokenizer.from_file(tokenizer_file)

        # Read dataset from csv file
        self.data = pd.read_csv(dataset_file).values
        np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        query, x = self.data[index]

        image = read_image(os.path.join(self.image_path, x))

        _, h, w = image.shape
        factor = self.img_res / min(w, h)

        new_width = int(w * factor)
        new_height = int(h * factor)

        image = T.Resize((new_height, new_width))(image)

        # Apply any data transformations if specified
        if self.transform is not None:
            image = self.transform(image)

        # Tokenize query
        encoding = self.tokenizer.encode(query)
        token = torch.tensor(encoding.ids)

        return image, token
