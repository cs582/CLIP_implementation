import unittest
import torch
import time
from eval_loop.models.computer_vision.cv_utils import my_images_to_pathces_implementation as img_to_patches1
from eval_loop.models.computer_vision.cv_utils import optimized_images_to_patches_implementation as img_to_patches2
from eval_loop.models.computer_vision.cv_utils import eignops_images_to_patches_optimization as img_to_patches3


class UtilsTest(unittest.TestCase):
    def test_image_to_patches_1(self):
        p = 16
        c, h, w = 3, 128, 128
        batch_size = 64

        x = torch.rand(batch_size, c, h, w)
        n_rows = h // p
        n_cols = w // p

        n = n_rows*n_cols
        start = time.time()
        patches = img_to_patches1(x, batch_size, c, h, w, p)
        end = time.time()

        message = f"Mine: time {end - start} seconds"
        print(message)

        self.assertEqual(patches.shape, (batch_size, n, c*p*p))

    def test_image_to_patches_2(self):
        p = 16
        c, h, w = 3, 128, 128
        batch_size = 64

        x = torch.rand(batch_size, c, h, w)
        n_rows = h // p
        n_cols = w // p

        n = n_rows*n_cols
        start = time.time()
        patches = img_to_patches2(x, batch_size, c, h, w, p)
        end = time.time()

        message = f"ChatGPT: time {end - start} seconds"
        print(message)

        self.assertEqual(patches.shape, (batch_size, n, c*p*p), msg=message)

    def test_image_to_patches_3(self):
        p = 16
        c, h, w = 3, 128, 128
        batch_size = 64

        x = torch.rand(batch_size, c, h, w)
        n_rows = h // p
        n_cols = w // p

        n = n_rows*n_cols
        start = time.time()
        patches = img_to_patches3(x, batch_size, c, h, w, p)
        end = time.time()

        message = f"Einops: time {end - start} seconds"
        print(message)

        self.assertEqual(patches.shape, (batch_size, n, c*p*p), msg=message)