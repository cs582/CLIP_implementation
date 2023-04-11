import unittest
import torch
import time

from src.models.computer_vision.cv_modules import BlurPool2d, Convolution1, Convolution2, Convolution3, Convolution4, Convolution5
from src.models.computer_vision.cv_backbones import RN34_at224


class ResnetModulesUnitTest(unittest.TestCase):
    def test_blur_pooling_2d(self):
        x = torch.rand(64, 3, 56, 56)
        model = BlurPool2d(n_channels=3)

        start = time.time()
        out = model(x)
        end = time.time()

        message = f"BlurPool forward time: {end - start} seconds"
        print(message)

        self.assertEqual(out.shape, (64, 3, 28, 28), msg=f"Failed, out size {out.shape} should be (64, 3, 28, 28)")

    def test_residual_1(self):
        x = torch.rand(32, 3, 224, 224)
        model = Convolution1()

        start = time.time()
        out = model(x)
        end = time.time()

        message = f"Convolution1 forward time: {end - start} seconds"
        print(message)

        self.assertEqual(out.shape, (32, 64, 56, 56), msg=f"Failed, out size {out.shape} should be (32, 64, 56, 56)")

    def test_residual_2(self):
        x = torch.rand(32, 3, 56, 56)
        model = Convolution2(n_channels=3)

        start = time.time()
        out = model(x)
        end = time.time()

        message = f"Convolution2 forward time: {end - start} seconds"
        print(message)

        self.assertEqual(out.shape, (32, 3, 56, 56), msg=f"Failed, out size {out.shape} should be (32, 3, 56, 56)")

    def test_residual_3(self):
        x = torch.rand(32, 3, 56, 56)
        model = Convolution3(in_channels=3, out_channels=64)

        start = time.time()
        out = model(x)
        end = time.time()

        message = f"Convolution3 forward time: {end - start} seconds"
        print(message)

        self.assertEqual(out.shape, (32, 64, 28, 28), msg=f"Failed, out size {out.shape} should be (32, 64, 28, 28)")

    def test_residual_4(self):
        x = torch.rand(32, 3, 28, 28)
        model = Convolution4(in_channels=3, out_channels=64)

        start = time.time()
        out = model(x)
        end = time.time()

        message = f"Convolution4 forward time: {end - start} seconds"
        print(message)

        self.assertEqual(out.shape, (32, 64, 14, 14), msg=f"Failed, out size {out.shape} should be (32, 64, 14, 14)")

    def test_residual_5(self):
        x = torch.rand(32, 3, 14, 14)
        model = Convolution5(in_channels=3, out_channels=64)

        start = time.time()
        out = model(x)
        end = time.time()

        message = f"Convolution5 forward time: {end - start} seconds"
        print(message)

        self.assertEqual(out.shape, (32, 64, 7, 7), msg=f"Failed, out size {out.shape} should be (32, 64, 7, 7)")


class BackbonesUnitTest(unittest.TestCase):
    def test_RN_at_224(self):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        x = torch.rand(2, 3, 224, 224).to(device)
        model = RN34_at224(embedding_dim=1000).to(device)

        start = time.time()
        out = model(x)
        end = time.time()

        message = f"RN@224 forward time: {end - start} seconds"
        print(message)

        self.assertEqual(out.shape, (2, 1000), msg=f"Failed, out size {out.shape} should be (2,1000)")

