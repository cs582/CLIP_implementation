import json
import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class ImageQueryDataset(Dataset):
    def __init__(self, data_dir, filename, img_res=(224, 224)):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.RandomCrop(img_res),
            transforms.ToTensor()
        ])

        # Get a list of image file paths
        with open(f"{data_dir}/{filename}", 'r') as f:
            self.image_files = json.load(f)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # Load the image at the given index
        image_path = self.image_files[index]
        image_full_path = f"{self.data_dir}/images/{image_path}"
        print(image_full_path)
        image = cv2.imread(image_full_path)
        print(image)

        # Apply any data transformations if specified
        if self.transform is not None:
            image = self.transform(image)

        # Get the label from the image file name
        label = image_path.replace('.jpg', '').replace('_', ' ')

        return image, label
