import unittest
import torch
import time
import numpy as np

from eval_loop.models.natural_language_processing.nlp_modules import MaskedSelfAttention, MaskedMultiHeadSelfAttention, TransformerRadford


class TransformerRadfordUnitTest(unittest.TestCase):
    def test_self_attention(self):
        token_size = 512
        max_length = 72

        # Reproducing random length sentences
        mask = torch.zeros(64, max_length).to(dtype=torch.bool)
        for i in range(64):
            mask[i, :np.random.randint(low=0, high=max_length)] = 1.0

        x = torch.rand(64, max_length, token_size)
        model = MaskedSelfAttention(dim_x=token_size, dim_att=128)

        start = time.time()
        out = model(x, mask)
        end = time.time()

        self.assertEqual(out.shape, (64, max_length, 128), msg=f"Wrong output shape, got {out.shape} should be (64, {max_length}, 128)")

        print(f"SelfAttention forward time: {end-start} seconds")

    def test_multihead_self_attention(self):
        token_size = 512
        max_length = 72

        # Reproducing random length sentences
        mask = torch.zeros(64, max_length).to(dtype=torch.bool)
        for i in range(64):
            mask[i, :np.random.randint(low=0, high=max_length)] = 1.0

        x = torch.rand(64, max_length, token_size)
        model = MaskedMultiHeadSelfAttention(dim_model=token_size, n_head=8)

        start = time.time()
        out = model(x, mask)
        end = time.time()

        self.assertEqual(out.shape, (64, max_length, token_size), msg=f"Wrong output shape, got {out.shape} should be (64, {max_length}, {token_size})")

        print(f"MultiHeadSelfAttention forward time: {end-start} seconds")

    def test_transformer_radford_layer(self):
        token_size = 512
        max_length = 72

        # Reproducing random length sentences
        mask = torch.zeros(64, max_length).to(dtype=torch.bool)
        for i in range(64):
            mask[i, :np.random.randint(low=0, high=max_length)] = 1.0

        x = torch.rand(64, max_length, token_size)
        model = TransformerRadford(dim_model=token_size, nhead=8, dim_ff=1024)

        start = time.time()
        out = model(x, mask)
        end = time.time()

        self.assertEqual(out.shape, (64, max_length, token_size), msg=f"Wrong output shape, got {out.shape} should be (64, {max_length}, {token_size})")

        print(f"Transformer Radford forward time: {end-start} seconds")


