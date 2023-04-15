import unittest
import torch
import time

import numpy as np

from src.models.natural_language_processing.nlp_backbones import TransformerB, TransformerL


class BackbonesTextUnitTest(unittest.TestCase):
    def test_transformer_decoder_base(self):
        n_batches = 128
        max_length = 25

        dim_out = 512

        x = torch.randint(low=0, high=1, size=(n_batches, max_length))
        for q_idx in range(len(x)):
            x[q_idx, np.random.randint(low=1, high=max_length):] = -1.0
        model = TransformerB(dim_out=dim_out, batch_size=n_batches, vocab_size=1000, max_length=max_length)

        start = time.time()
        out = model(x)
        end = time.time()

        message = f"Transformer Decoder Base forward time: {end - start} seconds"
        print(message)

        self.assertEqual(out.shape, (n_batches, dim_out), msg=f"Transformer Decoder Failed, out size {out.shape} should be ({n_batches}, {dim_out})")

    def test_transformer_decoder_large(self):
        batch_size = 128
        max_length = 25

        dim_out = 768

        x = torch.randint(low=0, high=1, size=(batch_size, max_length))
        for q_idx in range(len(x)):
            x[q_idx, np.random.randint(low=1, high=max_length):] = -1.0
        model = TransformerL(dim_out=dim_out, batch_size=batch_size, vocab_size=1000, max_length=max_length)

        start = time.time()
        out = model(x)
        end = time.time()

        message = f"Transformer Decoder Large forward time: {end - start} seconds"
        print(message)

        self.assertEqual(out.shape, (batch_size, dim_out), msg=f"Transformer Decoder Failed, out size {out.shape} should be ({batch_size}, {dim_out})")


class BackbonesTextGPUUnitTest(unittest.TestCase):
    def test_transformer_decoder_base(self):
        device = torch.device('cuda:0')

        n_batches = 128
        max_length = 25

        dim_out = 512

        x = torch.randint(low=0, high=1, size=(n_batches, max_length)).to(device)
        for q_idx in range(len(x)):
            x[q_idx, np.random.randint(low=1, high=max_length):] = -1.0
        model = TransformerB(dim_out=dim_out, batch_size=n_batches, vocab_size=1000, max_length=max_length).to(device)

        start = time.time()
        out = model(x)
        end = time.time()

        message = f"Transformer Decoder Base [CUDA: {torch.cuda.get_device_name(0)}] forward time: {end - start} seconds"
        print(message)

        self.assertEqual(out.shape, (n_batches, dim_out), msg=f"Transformer Decoder Failed, out size {out.shape} should be ({n_batches}, {dim_out})")

    def test_transformer_decoder_large(self):
        device = torch.device('cuda:0')

        n_batches = 128
        max_length = 25

        dim_out = 768

        x = torch.randint(low=0, high=1, size=(n_batches, max_length)).to(device)
        for q_idx in range(len(x)):
            x[q_idx, np.random.randint(low=1, high=max_length):] = -1.0
        model = TransformerL(dim_out=dim_out, batch_size=n_batches, vocab_size=1000, max_length=max_length).to(device)

        start = time.time()
        out = model(x)
        end = time.time()

        message = f"Transformer Decoder Large [CUDA: {torch.cuda.get_device_name(0)}] forward time: {end - start} seconds"
        print(message)

        self.assertEqual(out.shape, (n_batches, dim_out), msg=f"Transformer Decoder Failed, out size {out.shape} should be ({n_batches}, {dim_out})")
