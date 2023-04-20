import unittest
import torch
import time
import apex
import torch.nn as nn

import numpy as np

from src.models.CLIP_model import CLIPModule
from src.models.computer_vision.backbones.vit import ViTat224, ViTat336
from src.models.computer_vision.backbones.resnet34 import RN34at224, RN34at336
from src.models.natural_language_processing.nlp_backbones import TransformerB
from src.utils import CLIPLoss


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
        tokenized_words = torch.randint(low=1, high=vocab_size, size=(batch_size, max_length))
        for q_idx in range(len(tokenized_words)):
            tokenized_words[q_idx, np.random.randint(low=1, high=max_length):] = 0.0

        # Output
        start = time.time()
        out = clip_model(imgs, tokenized_words)
        end = time.time()

        # Assert case
        self.assertEqual(out.shape, (batch_size, batch_size))
        print(f"CLIP forward finished in {end-start} seconds with a batch of {batch_size}.")



class CLIPGPUUnitTest(unittest.TestCase):
    def test_clip_module_with_ViTat224_in_gpu(self):
        device = torch.device("cuda:0")

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
        clip_model = CLIPModule(image_encoder, text_encoder, dim_img, dim_text, embedding_dim, temperature).to(device)

        # Testing images
        resolution = 224
        imgs = torch.rand(batch_size, 3, resolution, resolution).to(device)
        tokenized_words = torch.randint(low=1, high=vocab_size, size=(batch_size, max_length)).to(device)
        for q_idx in range(len(tokenized_words)):
            tokenized_words[q_idx, np.random.randint(low=1, high=max_length):] = 0.0

        # Output
        start = time.time()
        out = clip_model(imgs, tokenized_words)
        end = time.time()

        # Assert case
        self.assertEqual(out.shape, (batch_size, batch_size))
        print(f"CLIP [CUDA: {torch.cuda.get_device_name(0)}] forward finished in {end-start} seconds with a batch of {batch_size}.")

    def test_backward_clip_module_with_ViTat224_in_gpu(self):
        device = torch.device("cuda:0")

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

        # Enable cublasLt for mixed-precision training
        torch.backends.cudnn.enabled = True

        # Initialize encoders
        image_encoder = ViTat224(dim_out=dim_img)
        text_encoder = TransformerB(dim_out=dim_text, batch_size=batch_size, vocab_size=vocab_size, max_length=max_length)

        # Initialize CLIP
        clip_model = CLIPModule(image_encoder, text_encoder, dim_img, dim_text, embedding_dim, temperature).to(device)
        optimizer = torch.optim.Adam(clip_model.parameters(), lr=2e-5)

        # Initialize loss function
        loss_function = CLIPLoss(batch_size).to(device)

        # Testing images
        resolution = 224
        imgs = torch.rand(batch_size, 3, resolution, resolution).to(device)
        tokenized_words = torch.randint(low=1, high=vocab_size, size=(batch_size, max_length)).to(device)
        for q_idx in range(len(tokenized_words)):
            tokenized_words[q_idx, np.random.randint(low=1, high=max_length):] = 0.0

        # Training process
        start_time = time.time()

        # Optim step
        optimizer.zero_grad()

        # Get cosine similarities
        logits = clip_model(imgs, tokenized_words)

        # Compute loss and backpropagation
        loss = loss_function(logits)
        loss.backward()

        # Take step
        optimizer.step()

        # End training
        end_time = time.time()

        # Assert case
        self.assertEqual(logits.shape, (batch_size, batch_size))
        self.assertNotEqual(loss.item(), torch.nan)
        print(f"CLIP [CUDA: {torch.cuda.get_device_name(0)}] backward finished in {end_time-start_time} seconds with a batch of {batch_size}.")
