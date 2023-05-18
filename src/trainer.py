import numpy as np
import boto3
import torch
import json
import os


from torch.cuda.amp import GradScaler
from tqdm import tqdm
from src.utils import save_checkpoint, load_from_checkpoint


s3 = boto3.client('s3')

# Models directory
models_dir = "src/models/checkpoints"


def training(training_dataset, clip_model, loss_function, optimizer, scheduler, epochs, device, model_name, load_last_checkpoint=False, load_from_given_checkpoint=None):
    """
    Training loop.
    :param training_dataset: (Dataloader) training data.
    :param clip_model: (torch.nn.Module) CLIP model.
    :param loss_function: (torch.nn.Module) CLIP loss function.
    :param optimizer: (torch.nn.Optimizer) Torch optimizer.
    :param scheduler: (object) Learning rate scheduler.
    :param epochs: (int) Number of epochs.
    :param device: (torch.device) Torch device.
    :param model_name: (str) Image encoder name.
    :param load_last_checkpoint: (bool) Load from the last epoch (default=False).
    :param load_from_given_checkpoint: (str) Gives it a specific model to load.
    :return:
    """
    history_filename = f"clip_loss_{model_name}.json"

    history_loss = []

    epoch_0 = 0
    if load_last_checkpoint:
        model_path = max([os.path.join(models_dir, x) for x in os.listdir(models_dir)], key=os.path.getctime)
        epoch_0, history_loss = load_from_checkpoint(model_path, clip_model, scheduler, optimizer)
        epoch_0 += 1

    elif load_from_given_checkpoint is not None:
        epoch_0, history_loss = load_from_checkpoint(load_from_given_checkpoint, clip_model, scheduler, optimizer)
        epoch_0 += 1

    # Initialize Gradient
    optimizer.zero_grad(set_to_none=True)
    scaler = GradScaler()

    # Every N batches
    accumulate = 64

    for epoch in range(epoch_0, epochs):
        pbar = tqdm(total=len(training_dataset))
        for idx, (images, queries) in enumerate(training_dataset):

            # Mixed Precision Forward Pass
            with torch.cuda.amp.autocast():
                # Extract feature representations
                logits_images, logits_text = clip_model(images, queries)

                # Compute Loss
                loss = loss_function(logits_images, logits_text)

            # Save to loss history
            history_loss.append(np.round(loss.item(), 5))
            last_lr = np.round(scheduler.get_last_lr(), 9)


            # Full Precision Back-propagation
            scaler.scale(loss).backward()

            # Set pbar description
            pbar.set_description(f"Epoch:{epoch}. Loss:{history_loss[-1]}. lr:{last_lr}")

            # Update every n batches
            if (idx+1) % accumulate == 0 or (idx+1) == len(training_dataset):
                # Optimization
                scaler.step(optimizer)

                # Take learning rate step
                scheduler.step()

                # Update scaler
                scaler.update()

                # Reset the gradients to None
                optimizer.zero_grad(set_to_none=True)

                # Update progress bar
                pbar.update(accumulate)

            # Save to S3
            if (idx+1) % 5000 == 0 or (idx+1) == len(training_dataset):
                history_bytes = json.dumps(history_loss)
                s3.put_object(Bucket='clip-loss-may-1', Key=history_filename, Body=history_bytes)

        # Save at every epoch
        save_checkpoint(model=clip_model, optimizer=optimizer, epoch=epoch, history=history_loss, models_dir=models_dir, scheduler=scheduler)

