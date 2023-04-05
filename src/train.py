import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import CLIPLoss

from models.computer_vision.cv_backbones import ViT
from models.natural_language_processing.nlp_backbones import Transformer


def training(training_dataset, image_encoder, text_encoder, temperature, optimizer):
    loss_func = CLIPLoss(temperature=temperature)

    for images, queries in training_dataset:
        image_projection = None
        text_projection = None

        # Extract feature representations
        images_features = image_encoder(images)
        texts_features = text_encoder(queries)

        # Multimodal embedding
        images_embeddings = F.normalize(torch.dot(images_features, image_projection), dim=1)
        texts_embeddings = F.normalize(torch.dot(texts_features, text_projection), dim=1)

        # Initialize Gradient
        optimizer.zero_grad()

        # Compute Loss
        loss = loss_func(images_embeddings, texts_embeddings)

        # Backpropagation
        loss.backward()

        # Optimization
        optimizer.step()

