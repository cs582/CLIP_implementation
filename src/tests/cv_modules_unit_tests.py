import unittest
import torch
import time

from src.models.computer_vision.cv_modules import BlurPool2d, Convolution1, Convolution2, Convolution3, Convolution4, Convolution5, AttentionPooling


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

    def test_attention_pooling(self):
        x = torch.rand(64, 512, 7, 7)
        model = AttentionPooling(in_size=(7,7), dim_attention=32)

        start = time.time()
        out = model(x)
        end = time.time()

        message = f"AttentionPooling forward time: {end - start} seconds"
        print(message)

        self.assertEqual(out.shape, (64, 512, 32))

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




