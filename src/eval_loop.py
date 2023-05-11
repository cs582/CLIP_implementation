import numpy as np
import boto3
import json
import os

from tqdm import tqdm
from src.utils import save_checkpoint, load_from_checkpoint


s3 = boto3.client('s3')

# Models directory
models_dir = "src/models/checkpoints"


def eval(eval_dataset, clip_model, loss_function, device, model_name, load_from_given_checkpoint=None):
    """
    Training loop.
    :param eval_dataset: (Dataloader) training data.
    :param clip_model: (torch.nn.Module) CLIP model.
    :param loss_function: (torch.nn.Module) CLIP loss function.
    :param device: (torch.device) Torch device.
    :param model_name: (str) Image encoder name.
    :param load_from_given_checkpoint: (str) Gives it a specific model to load.
    :return:
    """
    _, _ = load_from_checkpoint(load_from_given_checkpoint, clip_model)

    loss_eval = []

    pbar = tqdm(total=len(eval_dataset), desc="Eval. Loop.")
    for idx, (images, queries) in enumerate(eval_dataset):
        images, queries = images.to(device), queries.to(device)

        # Extract feature representations
        logits_images, logits_text = clip_model(images, queries)

        # Compute Loss
        loss = loss_function(logits_images, logits_text)
        loss_eval.append(loss.item())

        pbar.update(1)

    print(f"Eval Loop DONE! AVG Loss: {np.round(np.mean(loss_eval), 5)}.")