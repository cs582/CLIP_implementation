import unittest
import torch
import time
import numpy as np

from src.models.natural_language_processing.utils import ScaledDotProductAttention
from src.models.natural_language_processing.utils import MultiHeadSelfAttention
from src.models.natural_language_processing.utils import TransformerRadford


class TransformerRadfordUnitTest(unittest.TestCase):
    def test_scaled_dot_product_attention(self):
        # batch of 8 with 75 words of dim 25
        batch_size = 32

        q = torch.rand(batch_size, 75, 25)
        k = torch.rand(batch_size, 75, 25)
        v = torch.rand(batch_size, 75, 25)

        # Fake a mask of words
        mask = torch.ones(batch_size, 75, dtype=torch.bool)
        for w in range(0, 75):
            mask[:, np.random.randint(low=75//2, high=75):] = 0.0

        start = time.time()
        model = ScaledDotProductAttention()
        end = time.time()
        self.assertEqual(model(q, k, v, mask).shape, (batch_size, 75, 25), msg="Dot Product Attention Failed")

        print(f"Scaled Dot Product Attention Masked finished in {end-start} seconds")

    def test_multi_head_self_attention(self):
        # batch of 8 with 75 words embedded on 25 dim
        num_words = 75
        batch_size = 32
        embedding_dim = 25
        latent_vector_size = 512

        # nhead
        nhead = 8

        x = torch.rand(batch_size, num_words, embedding_dim)

        # Fake a mask of words
        mask = torch.ones(batch_size, num_words, dtype=torch.bool)
        for w in range(0, num_words):
            mask[:, np.random.randint(low=num_words//2, high=num_words):] = 0.0

        start = time.time()
        model = MultiHeadSelfAttention(embedd_dim=embedding_dim, vector_size=latent_vector_size, nhead=nhead)
        end = time.time()

        self.assertEqual(model(x, mask).shape, (batch_size, num_words, latent_vector_size), msg="Self Attention Failed")

        print(f"Multi-head Masked Self-Attention finished in {end-start} seconds")

    def test_transformer_radford(self):
        # batch of 8 with 75 words embedded on 25 dim
        num_words = 75
        batch_size = 32
        embedding_dim = 25
        forward_dim = 1024
        latent_vector_size = 512

        # nhead
        nhead = 8

        x = torch.rand(batch_size, num_words, embedding_dim)

        # Fake a mask of words
        mask = torch.ones(batch_size, num_words, dtype=torch.bool)
        for w in range(0, num_words):
            mask[:, np.random.randint(low=num_words//2, high=num_words):] = 0.0

        start = time.time()
        model = TransformerRadford(embedding_dim=embedding_dim, latent_vector_size=latent_vector_size, forward_dim=forward_dim, nhead=nhead)
        end = time.time()

        self.assertEqual(model(x, mask).shape, (batch_size, num_words, latent_vector_size), msg="Transformer Radford")

        print(f"Transformer Radford forward finished in {end-start} seconds")
