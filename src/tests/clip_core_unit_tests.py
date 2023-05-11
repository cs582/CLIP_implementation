import unittest
import torch
import time

from torch.cuda.amp import autocast, GradScaler

import numpy as np

from eval_loop.models.CLIP_model import CLIPModule
from eval_loop.models.computer_vision.backbones.vit import ViTat112, ViTat224
from eval_loop.models.natural_language_processing.nlp_backbones import TransformerB
from eval_loop.models.CLIP_Loss import CLIPLoss


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
    def test_backward_clip_module_with_ViTat224_in_gpu(self):
        device = torch.device("cuda:0")

        # CLIP parameters
        batch_size = 64
        embedding_dim = 512
        temperature = 0.07

        # Text Encoder parameters
        vocab_size = 15000
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

        # Define the scaler for mixed precision training
        scaler = GradScaler()

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

        with autocast():
            # Get cosine similarities
            logits = clip_model(imgs, tokenized_words)
            # Compute loss and backpropagation
            loss = loss_function(logits)

        # Scale the loss and perform the backward pass with autocasting
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Take step
        optimizer.step()

        # End training
        end_time = time.time()

        # Assert case
        self.assertEqual(logits.shape, (batch_size, batch_size), msg=f"logits shape {logits.shape} doesn't match expected size ({batch_size}, {batch_size})")
        self.assertNotEqual(loss.item(), torch.nan, msg=f"Loss item is {loss.item()}")
        print(f"CLIP ViT@224 [CUDA: {torch.cuda.get_device_name(0)}] backward finished in {end_time-start_time} seconds with a batch of {batch_size}.")

    def test_backward_clip_module_with_ViTat112_in_gpu(self):
        device = torch.device("cuda:0")

        # CLIP parameters
        batch_size = 128
        embedding_dim = 512
        temperature = 0.07

        # Text Encoder parameters
        vocab_size = 15000
        max_length = 24
        dim_text = 256

        # Image Encoder parameters
        dim_img = 384

        # Enable cublasLt for mixed-precision training
        torch.backends.cudnn.enabled = True

        # Initialize encoders
        image_encoder = ViTat112(dim_out=dim_img)
        text_encoder = TransformerB(dim_out=dim_text, batch_size=batch_size, vocab_size=vocab_size, max_length=max_length)

        # Initialize CLIP
        clip_model = CLIPModule(image_encoder, text_encoder, dim_img, dim_text, embedding_dim, temperature).to(device)
        optimizer = torch.optim.Adam(clip_model.parameters(), lr=2e-5)

        # Define the scaler for mixed precision training
        scaler = GradScaler()

        # Initialize loss function
        loss_function = CLIPLoss(batch_size).to(device)

        # Testing images
        resolution = 112
        imgs = torch.rand(batch_size, 3, resolution, resolution).to(device)
        tokenized_words = torch.randint(low=1, high=vocab_size, size=(batch_size, max_length)).to(device)
        for q_idx in range(len(tokenized_words)):
            tokenized_words[q_idx, np.random.randint(low=1, high=max_length):] = 0.0

        # Training process
        start_time = time.time()

        # Optim step
        optimizer.zero_grad()

        with autocast():
            # Get cosine similarities
            logits = clip_model(imgs, tokenized_words)
            # Compute loss and backpropagation
            loss = loss_function(logits)

        # Scale the loss and perform the backward pass with autocasting
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Take step
        optimizer.step()

        # End training
        end_time = time.time()

        # Assert case
        self.assertEqual(logits.shape, (batch_size, batch_size), msg=f"logits shape {logits.shape} doesn't match expected size ({batch_size}, {batch_size})")
        self.assertNotEqual(loss.item(), torch.nan, msg=f"Loss item is {loss.item()}")
        print(f"CLIP ViT@112 [CUDA: {torch.cuda.get_device_name(0)}] backward finished in {end_time-start_time} seconds with a batch of {batch_size}.")
