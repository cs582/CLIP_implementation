import unittest
import torch
import time

from src.models.natural_language_processing.utils import ScaledDotProductAttention
from src.models.natural_language_processing.utils import MultiHeadSelfAttention


class TransformerRadfordUnitTest(unittest.TestCase):
    def test_scaled_dot_product_attention(self):
        # batch of 8 with 75 words of dim 25
        q = torch.rand(8, 75, 25)
        k = torch.rand(8, 75, 25)
        v = torch.rand(8, 75, 25)

        mask = torch.rand(8, 25, 25) < 0.1

        start = time.time()
        model = ScaledDotProductAttention(dim=25)
        end = time.time()
        self.assertEqual(model(q, k, v, mask).shape, (8, 25, 75))

        print(f"Scaled Dot Product Attention Masked finished in {end-start} seconds")

    def test_multi_head_self_attention(self):
        # batch of 8 with 75 words of dim 25
        q = torch.rand(8, 75, 25)
        k = torch.rand(8, 75, 25)
        v = torch.rand(8, 75, 25)

        mask = torch.rand(8, 25) < 0.1

        start = time.time()
        model = MultiHeadSelfAttention(dim=64, nhead=8)
        end = time.time()

        self.assertEqual(model(q, k, v, mask).shape, (8, 75, 68))

        print(f"Multi-head Masked Self-Attention finished in {end-start} seconds")
