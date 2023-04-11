import unittest
import torch
import time

from src.models.computer_vision.backbones.resnet34 import RN34at224, RN34at336


class BackbonesUnitTest(unittest.TestCase):
    def test_RN_at_224(self):
        x = torch.rand(4, 3, 224, 224)
        model = RN34at224(embedding_dim=1000)

        start = time.time()
        out = model(x)
        end = time.time()

        message = f"RN@224 forward time: {end - start} seconds"
        print(message)

        self.assertEqual(out.shape, (4, 1000), msg=f"RN@224 Failed, out size {out.shape} should be (2,1000)")

    def test_RN_at_336(self):
        x = torch.rand(4, 3, 336, 336)
        model = RN34at336(embedding_dim=1000)

        start = time.time()
        out = model(x)
        end = time.time()

        message = f"RN@336 forward time: {end - start} seconds"
        print(message)

        self.assertEqual(out.shape, (4, 1000), msg=f"RN@336 Failed, out size {out.shape} should be (2,1000)")


class BackbonesUnitTestGPU(unittest.TestCase):
    def test_RN_at_224(self):
        device = torch.device('cuda:0')

        x = torch.rand(128, 3, 224, 224).to(device)
        model = RN34at224(embedding_dim=1000).to(device).eval()

        start = time.time()
        out = model(x)
        end = time.time()

        message = f"RN@224 [CUDA: {torch.cuda.get_device_name(0)}] forward time: {end - start} seconds"
        print(message)

        self.assertEqual(out.shape, (4, 1000), msg=f"RN@224 Failed, out size {out.shape} should be (2,1000)")

    def test_RN_at_336(self):
        device = torch.device('cuda:0')

        x = torch.rand(128, 3, 336, 336).to(device)
        model = RN34at336(embedding_dim=1000).to(device).eval()

        start = time.time()
        out = model(x)
        end = time.time()

        message = f"RN@336 [CUDA: {torch.cuda.get_device_name(0)}] forward time: {end - start} seconds"
        print(message)

        self.assertEqual(out.shape, (4, 1000), msg=f"RN@336 Failed, out size {out.shape} should be (2,1000)")