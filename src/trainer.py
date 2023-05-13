import numpy as np
import boto3
import json
import os

from tqdm import tqdm
from src.utils import save_checkpoint, load_from_checkpoint


s3 = boto3.client('s3')

# Models directory
models_dir = "src/models/checkpoints"


def training(training_dataset, clip_model, loss_function, optimizer, scheduler, epochs, max_steps, device, model_name, load_last_checkpoint=False, load_from_given_checkpoint=None):
    """
    Training loop.
    :param training_dataset: (Dataloader) training data.
    :param clip_model: (torch.nn.Module) CLIP model.
    :param loss_function: (torch.nn.Module) CLIP loss function.
    :param optimizer: (torch.nn.Optimizer) Torch optimizer.
    :param scheduler: (object) Learning rate scheduler.
    :param epochs: (int) Number of epochs.
    :param max_steps: (int) Maximum number of steps.
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

    for epoch in range(epoch_0, epochs):
        batch_length = min(len(training_dataset), max_steps - len(history_loss))
        pbar = tqdm(total=batch_length)
        for idx, (images, queries) in enumerate(training_dataset):
            images, queries = images.to(device), queries.to(device)

            # Extract feature representations
            logits_images, logits_text = clip_model(images, queries)

            # Initialize Gradient
            optimizer.zero_grad()

            # Compute Loss
            loss = loss_function(logits_images, logits_text)

            # Save to loss history
            history_loss.append(np.round(loss.item(), 5))
            last_lr = np.round(scheduler.get_last_lr(), 9)

            # Backpropagation
            loss.backward()

            # Set pbar description
            pbar.set_description(f"Epoch:{epoch}. Loss:{history_loss[-1]}. lr:{last_lr}")

            # Optimization
            optimizer.step()
            scheduler.step()

            pbar.update(1)

            # Save to S3
            if (idx+1) % 2000 == 0 or (idx+1) == batch_length:
                history_bytes = json.dumps(history_loss)
                s3.put_object(Bucket='clip-loss-may-1', Key=history_filename, Body=history_bytes)

            if len(history_loss) >= max_steps:
                print("DONE!!!")
                save_checkpoint(model=clip_model, optimizer=optimizer, epoch=epoch, history=history_loss, models_dir=models_dir, scheduler=scheduler)
                break

        if len(history_loss) >= max_steps:
            break

        save_checkpoint(model=clip_model, optimizer=optimizer, epoch=epoch, history=history_loss, models_dir=models_dir, scheduler=scheduler)


