import unittest
import torch
import time

from src.models.computer_vision.cv_backbones import RN34_at224, RN34_at336


class BackbonesUnitTest(unittest.TestCase):
    def test_RN_at_224(self):
        x = torch.rand(4, 3, 224, 224)
        model = RN34_at224(embedding_dim=1000)

        start = time.time()
        out = model(x)
        end = time.time()

        message = f"RN@224 forward time: {end - start} seconds"
        print(message)

        self.assertEqual(out.shape, (4, 1000), msg=f"RN@224 Failed, out size {out.shape} should be (2,1000)")

    def test_RN_at_336(self):
        x = torch.rand(4, 3, 336, 336)
        model = RN34_at336(embedding_dim=1000)

        start = time.time()
        out = model(x)
        end = time.time()

        message = f"RN@336 forward time: {end - start} seconds"
        print(message)

        self.assertEqual(out.shape, (4, 1000), msg=f"RN@336 Failed, out size {out.shape} should be (2,1000)")