import os
import cv2
import torch
import time
import unittest
import numpy as np

from PIL import Image
from torchvision.transforms import transforms

transformation = transforms.Compose([transforms.ToTensor()])


class LoadingImagesUnitTest(unittest.TestCase):
    def test_pillow(self):
        img_path = "src/data/image_gen/WQ-dataset/images"

        times = [0.0] * 10

        for i, img_name in enumerate(range(10, 20)):
            img_filename = os.path.join(img_path, f"{img_name}.jpg")

            start = time.time()
            x = transformation(np.asarray(Image.open(img_filename)))
            end = time.time()

            times[i] = end - start

        print(f"Loaded images with PIL, avg time: {np.mean(times[1:])} x {x.shape}")

    def test_cv2(self):
        img_path = "src/data/image_gen/WQ-dataset/images"

        times = [0.0] * 10

        for i, img_name in enumerate(range(10, 20)):
            img_filename = os.path.join(img_path, f"{img_name}.jpg")

            start = time.time()
            x = transformation(cv2.imread(img_filename))
            end = time.time()

            times[i] = end - start

        print(f"Loaded images with OpenCV, avg time: {np.mean(times[1:])}. x {x.shape}")


