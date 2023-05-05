import numpy as np
import boto3
import json
import os

from tqdm import tqdm
from src.utils import save_checkpoint, load_from_checkpoint


s3 = boto3.client('s3')

# Models directory
models_dir = "src/models/checkpoints"


def training(training_dataset, clip_model, loss_function, optimizer, scheduler, epochs, device, model_name, load_last_checkpoint=False, load_from_given_checkpoint=None):
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
        pbar = tqdm(total=len(training_dataset))
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
            last_lr = np.round(scheduler.get_last_lr(), 5)

            # Backpropagation
            loss.backward()

            # Set pbar description
            pbar.set_description(f"Epoch:{epoch}. Loss:{history_loss[-1]}. lr:{last_lr}")

            # Optimization
            optimizer.step()
            scheduler.step()

            # Save to S3
            if (idx+1) % 2000 == 0:
                history_bytes = json.dumps(history_loss)
                s3.put_object(Bucket='clip-loss-may-1', Key=history_filename, Body=history_bytes)

            pbar.update(1)

        save_checkpoint(model=clip_model, optimizer=optimizer, epoch=epoch, history=history_loss, models_dir=models_dir, scheduler=scheduler)

