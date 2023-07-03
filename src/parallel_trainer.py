import numpy as np
import torch
import tensorboardX


from torch.cuda.amp import GradScaler
from tqdm import tqdm
from src.utils import save_checkpoint, warmup_scheduler

# Models directory
models_dir = "src/models/checkpoints"

def parallel_training(training_dataset, clip_models, losses, optimizers, epochs, lr_max, warmup_steps, max_steps, model_names):
    """
    Training loop.
    :param training_dataset: (Dataloader) training data.
    :param clip_models: (torch.nn.Module) CLIP model.
    :param losses: (list(torch.nn.Module)) list of CLIP loss functions.
    :param optimizers: (list(torch.nn.Optimizer)) list of Torch optimizers.
    :param epochs: (int) Number of epochs.
    :param lr_max: (float) maximum learning rate using warmup.
    :param warmup_steps: (int) warmup steps.
    :param max_steps: (int) maximum number of steps.
    :param model_names: (list(str)) Image encoder names.
    :return:
    """
    # Tensorboard writer
    writer = tensorboardX.SummaryWriter()

    # Store history loss
    history_loss = {
        "model_0": [],
        "model_1": [],
        "model_2": []
    }

    # Get models
    clip_model_0, clip_model_1, clip_model_2 = clip_models

    # Initialize Optimizer
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)

    # Create a list of schedulers
    schedulers = [warmup_scheduler(optimizer, warmup_steps=warmup_steps, warmup_start=0.0, lr_max=lr_max, max_steps=max_steps) for optimizer in optimizers]

    # Initialize Scaler
    scaler = GradScaler()

    # Wrap models with DataParallel
    clip_model_0 = torch.nn.DataParallel(clip_model_0, device_ids=[0])
    clip_model_1 = torch.nn.DataParallel(clip_model_1, device_ids=[1])
    clip_model_2 = torch.nn.DataParallel(clip_model_2, device_ids=[2])

    # Create a list of models and their respective optimizers
    models = [clip_model_0, clip_model_1, clip_model_2]


    for epoch in range(epochs):
        pbar = tqdm(total=len(training_dataset))
        for idx, (images, queries) in enumerate(training_dataset):
            # Load data
            image_batches = [
                images.to('cuda:0', non_blocking=True),
                images.to('cuda:1', non_blocking=True),
                images.to('cuda:2', non_blocking=True)
            ]
            query_batches = [
                queries.to('cuda:0', non_blocking=True),
                queries.to('cuda:1', non_blocking=True),
                queries.to('cuda:2', non_blocking=True)
            ]

            # Mixed Precision Forward Pass
            with torch.cuda.amp.autocast():
                # Extract feature representations for each model
                logits_images = []
                logits_text = []
                for model, images_model, queries_model in zip(models, image_batches, query_batches):
                    logits_image, logits_t = model(images_model, queries_model)
                    logits_images.append(logits_image)
                    logits_text.append(logits_t)

                # Compute Loss for each model
                losses = [criterion(logits_img, logits_t) for criterion, logits_img, logits_t in zip(losses, logits_images, logits_text)]

            # Save to loss history
            for i, loss in enumerate(losses):
                history_loss[f'model_{i}'].append(np.round(loss.item(), 5))

            # Full Precision Back-propagation and Optimization for each model
            for i, (loss, optimizer) in enumerate(zip(losses, optimizers)):
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # Take learning rate step for each scheduler
            for scheduler in schedulers:
                scheduler.step()
                last_lr = np.round(scheduler.get_last_lr(), 9)

            # Reset the gradients to None for each model
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)

            # Update progress bar
            pbar.update(1)
            pbar.set_description(f"lr: {last_lr}, loss [1: {history_loss[f'model_0'][-1]}] [2: {history_loss[f'model_1'][-1]}] [3: {history_loss[f'model_2'][-1]}")

            # See Training in Tensorboard
            for i, model in enumerate(models):
                writer.add_scalar(f'{model_names}_{i+1} Loss', history_loss[f'model_{i}'][-1], len(history_loss[f'model_{i}']))

        # Save at every epoch for each model
        for i, model in enumerate(models):
            save_checkpoint(model=model, optimizer=optimizers[i], epoch=epoch, history=history_loss[f'model_{i}'], models_dir=models_dir, scheduler=schedulers[i])
