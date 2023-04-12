import unittest
import torch
import time

from src.models.natural_language_processing.nlp_backbones import TextTransformer


class BackbonesTextUnitTest(unittest.TestCase):
    def test_transformer_decoder(self):
        n_batches = 128
        max_length = 72

        sentence_length = 74
        token_dim = 512

        dim_ff = 1024
        nhead = 8
        layers = 12

        x = torch.rand(n_batches, sentence_length, token_dim)
        model = TextTransformer(dim_model=token_dim, dim_ff=dim_ff, nhead=nhead, layers=layers, max_length=max_length, n_classes=1000)

        start = time.time()
        out = model(x)
        end = time.time()

        message = f"Transformer Decoder forward time: {end - start} seconds"
        print(message)

        self.assertEqual(out.shape, (n_batches, 1000), msg=f"Transformer Decoder Failed, out size {out.shape} should be ({n_batches}, 1000)")


class BackbonesTextGPUUnitTest(unittest.TestCase):
    def test_transformer_decoder(self):
        n_batches = 128

        sentence_length = 74
        token_dim = 512

        dim_ff = 1024
        nhead = 8
        layers = 12

        device = torch.device('cuda:0')

        x = torch.rand(n_batches, sentence_length, token_dim).to(device)
        model = TextTransformer(dim_model=token_dim, dim_ff=dim_ff, nhead=nhead, layers=layers, n_classes=1000).to(device)

        start = time.time()
        out = model(x)
        end = time.time()

        message = f"Transformer Decoder forward time: {end - start} seconds"
        print(message)

        self.assertEqual(out.shape, (n_batches, 1000), msg=f"Transformer Decoder Failed, out size {out.shape} should be ({n_batches}, 1000)")
