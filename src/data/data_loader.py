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
    def __init__(self, dataset_file, image_path, tokenizer_file, max_length, img_res=112):
        print("Initializing Dataset...")
        # Initial parameters
        self.img_res = img_res
        self.max_length = max_length

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

        print("Dataset Initialization DONE.")

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

        # Encode sequence
        encoded_query = self.tokenizer.encode(query).ids

        # Truncate query if necessary
        encoded_query = encoded_query[:self.max_length-2]

        # Add end_of_sentence token [EOS]
        encoded_query += [self.tokenizer.token_to_id('[EOS]')]

        # Add padding to encoded sentence
        encoded_query += [0] * (self.max_length - 2 - len(encoded_query))

        # Add [SOS] and [EOS] tokens
        encoded_query = [self.tokenizer.token_to_id('[SOS]')] + encoded_query

        # Add
        token = torch.tensor(encoded_query)

        return image, token
