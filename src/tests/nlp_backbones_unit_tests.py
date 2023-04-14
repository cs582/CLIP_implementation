import unittest
import torch
import time

import numpy as np

from src.models.natural_language_processing.nlp_backbones import TransformerB


class BackbonesTextUnitTest(unittest.TestCase):
    def test_transformer_decoder(self):
        n_batches = 128
        max_length = 74

        dim_out = 512

        mask = torch.zeros(n_batches, max_length).to(dtype=torch.bool)
        for i in range(n_batches):
            mask[i, :np.random.randint(low=1, high=max_length)] = 1.0

        x = torch.randint(low=0, high=1000, size=(n_batches, max_length))
        model = TransformerB(dim_out=dim_out, vocab_size=1000, max_length=max_length)

        start = time.time()
        out = model(x, mask)
        end = time.time()

        message = f"Transformer Decoder forward time: {end - start} seconds"
        print(message)

        self.assertEqual(out.shape, (n_batches, dim_out), msg=f"Transformer Decoder Failed, out size {out.shape} should be ({n_batches}, {dim_out})")


class BackbonesTextGPUUnitTest(unittest.TestCase):
    def test_transformer_decoder(self):
        device = torch.device('cuda:0')

        n_batches = 128
        max_length = 74

        dim_out = 512

        mask = torch.zeros(n_batches, max_length).to(device, dtype=torch.bool)
        for i in range(n_batches):
            mask[i, :np.random.randint(low=1, high=max_length)] = 1.0

        x = torch.randint(low=0, high=1000, size=(n_batches, max_length)).to(device)
        model = TransformerB(dim_out=dim_out, vocab_size=1000, max_length=max_length).to(device)

        start = time.time()
        out = model(x, mask)
        end = time.time()

        message = f"Transformer Decoder [CUDA: {torch.cuda.get_device_name(0)}] forward time: {end - start} seconds"
        print(message)

        self.assertEqual(out.shape, (n_batches, dim_out), msg=f"Transformer Decoder Failed, out size {out.shape} should be ({n_batches}, {dim_out})")