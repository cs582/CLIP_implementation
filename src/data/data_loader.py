import os
import cv2
import pandas as pd
import torch.utils.data
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def resize_image(image, h, w):
    resized_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized_image


class ImageQueryDataset(Dataset):
    def __init__(self, image_dir, csv_filepath, tokenizer, img_resolution=(224, 224)):

        # Initial parameters
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(img_resolution),
        ])
        self.img_resolution = img_resolution
        self.tokenizer = tokenizer

        # ->> IMAGE TRANSFORMATIONS <<- #
        images_names = [x for x in os.listdir(csv_filepath)]
        all_queries = pd.read_csv(csv_filepath, index_col=0, usecols=['query'])
        imgs = [os.path.join(image_dir, img) for img in images_names]

        # ->> TEXT TRANSFORMATIONS <<- #
        queries_indexes = all_queries.index[[int(x[:-4]) for x in images_names]]
        queries = all_queries[queries_indexes].tolist()

        # Check that queries have the same size as images
        n_queries, n_images = len(queries), len(imgs)
        assert n_queries == n_images, f"Exception. Queries and Images do not correspond in length. Got {n_queries} queries and {n_images} images."

        self.pairs = [[q, im] for q, im in zip(queries, imgs)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        # Get the current query
        query, image_path = self.pairs[index]

        # Load image with OpenCV
        img = cv2.imread(image_path)[:, :, ::-1]

        # Get image original dimensions
        img_h, img_w, channels = img.shape

        # Get current image resolution
        min_h, min_w = self.img_resolution

        # Placeholder for new image dimensions
        new_img_h, new_img_w = None, None

        # If the image is too big or too small, resize the smallest side to 1.2 of the minimum size
        if img_h > min_h * 1.5 or img_w > min_w * 1.5 or img_h < min_h or img_w < min_w:
            if img_h > img_w:
                gamma = (min_w * 1.2) / img_w
                new_img_h, new_img_w = min_h * gamma, min_w
            else:
                gamma = (min_h * 1.2) / img_h
                new_img_h, new_img_w = min_h, min_w * gamma

        # Resize image
        image = resize_image(img, new_img_h, new_img_w)

        # Apply any data transformations if specified
        if self.transform is not None:
            image = self.transform(image)

        # Tokenize query
        token = torch.tensor(self.tokenizer.tokenize(query))

        return image, token
