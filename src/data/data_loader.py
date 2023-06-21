import os

import numpy as np
import pandas as pd
import torch.utils.data

from tokenizers import Tokenizer
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as T


class ImageQueryDataset(Dataset):
    def __init__(self, dataset_file, image_path, tokenizer_file, max_length, start_from=None, end_at=None, img_res=112):
        """
        ImageQueryDataset Dataset object. This object initializes the dataset for the training loop.
        :param dataset_file: (str) Path dataset csv.
        :param image_path: (str) Path to images.
        :param tokenizer_file: (str) Path to tokenizer.
        :param max_length: (int) Maximum length of a sentence.
        :param img_res: (int) Image resolution, receives a single integer, will be used for both dimensions (img_res, img_res).
        """
        print("Initializing Dataset...")
        # Initial parameters
        self.img_res = img_res
        self.max_length = max_length
        self.last_index = None

        # Set image path
        self.image_path = image_path

        # Transformation is only Random Crop to match image resolution
        self.transform = T.Compose([
            T.RandomCrop((img_res, img_res)),
        ])

        # Load tokenizer from file
        self.tokenizer = Tokenizer.from_file(tokenizer_file)

        # Read dataset from csv file
        self.data = pd.read_csv(dataset_file).values[start_from:end_at]

        # Get first and last index
        first_idx = start_from if start_from is not None else 0
        last_idx = end_at if end_at is not None else len(self.data)

        # Shuffle data
        np.random.shuffle(self.data[first_idx: last_idx])

        print("Dataset Initialization DONE.")

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        query, x = self.data[index]

        image = read_image(os.path.join(self.image_path, x)).to(dtype=torch.float32)

        _, h, w = image.shape
        factor = self.img_res / min(w, h)

        new_width = int(w * factor) + 1
        new_height = int(h * factor) + 1

        image = T.Resize((new_height, new_width))(image)

        # Apply any data transformations if specified
        if self.transform is not None:
            try:
                image = self.transform(image)
            except:
                print(f"{image.shape}")

        # Encode sequence
        encoded_query = self.tokenizer.encode(query).ids

        # Truncate query if necessary
        encoded_query = encoded_query[:self.max_length-2]

        # Add end_of_sentence token [EOS]
        encoded_query += [self.tokenizer.token_to_id('[EOS]')]

        # Add padding to encoded sentence
        encoded_query += [0] * (self.max_length - len(encoded_query) - 1)

        # Add [SOS] and [EOS] tokens
        encoded_query = [self.tokenizer.token_to_id('[SOS]')] + encoded_query

        # Add
        token = torch.tensor(encoded_query, dtype=int)

        return image, token
