import unittest
import torch
import time

import numpy as np

from src.models.CLIP_model import CLIPModule
from src.models.computer_vision.backbones.vit import ViTat224, ViTat336
from src.models.computer_vision.backbones.resnet34 import RN34at224, RN34at336
from src.models.natural_language_processing.nlp_backbones import TransformerB


class CLIPUnitTest(unittest.TestCase):
    def test_clip_module_with_ViTat224_in_cpu(self):
        # CLIP parameters
        batch_size = 8
        embedding_dim = 512
        temperature = 0.07

        # Text Encoder parameters
        vocab_size = 32000
        max_length = 24
        dim_text = 512

        # Image Encoder parameters
        dim_img = 768

        # Initialize encoders
        image_encoder = ViTat224(dim_out=dim_img)
        text_encoder = TransformerB(dim_out=dim_text, batch_size=batch_size, vocab_size=vocab_size, max_length=max_length)

        # Initialize CLIP
        clip_model = CLIPModule(image_encoder, text_encoder, dim_img, dim_text, embedding_dim, temperature)

        # Testing images
        resolution = 224
        imgs = torch.rand(batch_size, 3, resolution, resolution)
        tokenized_words = torch.randint(low=0, high=1, size=(batch_size, max_length))
        for q_idx in range(len(tokenized_words)):
            tokenized_words[q_idx, np.random.randint(low=1, high=max_length):] = -1.0

        # Output
        start = time.time()
        out = clip_model(imgs, tokenized_words)
        end = time.time()

        # Assert case
        self.assertEqual(out.shape, (batch_size, batch_size))
        print(f"CLIP forward finished in {end-start} seconds with a batch of {batch_size}.")






