import torch
import torch.nn.functional as F

from src.models.natural_language_processing.nlp_backbones import TransformerB, TransformerL
from src.models.computer_vision.backbones.resnet34 import RN34at224, RN34at336
from src.models.computer_vision.backbones.vit import ViTat224, ViTat336

from src.models.CLIP_model import CLIPModule

from utils import CLIPLoss


def training(training_dataset, image_encoder, text_encoder, temperature, optimizer, epochs, embedding_dim):
    loss_func = CLIPLoss(logits_length=embedding_dim)
    model = CLIPModule(image_encoder=image_encoder, text_encoder=text_encoder, dim_img='###', dim_text='###', number_of_pairs=embedding_dim, temperature=temperature)

    for epoch in range(0, epochs):
        for images, queries in training_dataset:
            # Extract feature representations
            logits = model(images, queries)

            # Initialize Gradient
            optimizer.zero_grad()

            # Compute Loss
            loss = loss_func(logits)

            # Backpropagation
            loss.backward()

            # Optimization
            optimizer.step()

