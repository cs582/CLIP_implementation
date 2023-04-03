import torch
import torch.nn.functional as F


def CLIP(images, texts, image_encoder, text_encoder, image_projection, text_projection, temperature, loss_function):

    # Extract feature representations
    images_features = image_encoder(images)
    texts_features = text_encoder(texts)

    # Multimodal embedding
    images_embeddings = F.normalize(torch.dot(images_features, image_projection), dim=1)
    texts_embeddings = F.normalize(torch.dot(texts_features, text_projection), dim=1)

    # Scaled pairwise cosine similarities
    logits = torch.dot(images_embeddings, texts_embeddings) * torch.exp(temperature)

    # Loss function
    labels = torch.arange(0, len(images))
    loss_images = loss_function(logits, labels)
    loss_text = loss_function(logits.T, labels)
    loss = (loss_images + loss_text)/2
