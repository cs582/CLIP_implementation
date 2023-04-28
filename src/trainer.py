from tqdm import tqdm
from src.utils import save_checkpoint, load_from_checkpoint
import numpy as np
import os

# Models directory
models_dir = "src/models/checkpoints"


def training(training_dataset, clip_model, loss_function, optimizer, scheduler, epochs, device, load_last_checkpoint=False, load_from_given_checkpoint=None):
    loss_history = []

    epoch_0 = 0
    if load_last_checkpoint:
        model_path = max([os.path.join(models_dir, x) for x in os.listdir(models_dir)], key=os.path.getctime)
        epoch_0, loss_history = load_from_checkpoint(model_path, clip_model, optimizer)
    elif load_from_given_checkpoint is not None:
        epoch_0, loss_history = load_from_checkpoint(load_from_given_checkpoint, clip_model, optimizer)

    for epoch in range(epoch_0, epochs):
        # Taking 100 steps for fine-tuning
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
            loss_history.append(np.round(loss.item(), 5))

            # Backpropagation
            loss.backward()

            # Optimization
            optimizer.step()
            scheduler.step()

            pbar.set_description(f"Epoch:{epoch}. CURR LOSS:{loss_history[-1]}")
            pbar.update(1)

        save_checkpoint(model=clip_model, optimizer=optimizer, epoch=epoch, loss_history=loss_history, models_dir=models_dir)

