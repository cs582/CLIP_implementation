import os
import torch
from torch.optim.lr_scheduler import LambdaLR
from datetime import datetime as dt


def warmup_scheduler(optimizer, warmup_steps, warmup_start, lr_max, max_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return warmup_start + step * (lr_max - warmup_start) / warmup_steps
        elif step < max_steps:
            return warmup_start - step * (lr_max - warmup_start) / (max_steps - warmup_steps)
        else:
            return warmup_start

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(model, optimizer, epoch, scheduler, history, models_dir):
    """
    Save torch model checkpoint.

    :param model: (torch.nn.Module) Torch model.
    :param optimizer: (torch.nn.Optimizer) Torch optimizer.
    :param epoch: (int) Last epoch trained on.
    :param scheduler: (object) Learning rate scheduler object for warmup.
    :param history: (list) History loss.
    :param models_dir: (str) Train models directory.
    :return: None
    """
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    PATH = os.path.join(models_dir, f"CLIP_epoch_{epoch}_{dt.strftime(dt.now(), '%Y-%m-%d_%H:%M:%S')}")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': history,
        'scheduler': scheduler.state_dict()
    }, PATH)

    print(f"CLIP saved as {PATH}")


def load_from_checkpoint(model_filepath, model, scheduler=None, optimizer=None):
    """
    Load torch model from given path.

    :param model_filepath: (str) model filepath.
    :param model: (torch.nn.Module) torch backbone.
    :param scheduler: (object) learning rate optimizer.
    :param optimizer: (torch.nn.Optimizer) torch optimizer.
    :return:
    """

    checkpoint = torch.load(model_filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss_history = checkpoint['loss_history']

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    print(f"CLIP successfully loaded from {model_filepath}")

    return epoch, loss_history


def training_info_log_message(device, use_checkpoint, epochs, max_steps, accumulate, batch_size, image_encoder, text_encoder, image_dim_out, text_dim_out, optimizer):
    """
    Prints the training loop information.

    :param device: (torch.device) device.
    :param epochs: (int) number of epochs.
    :param max_steps: (int) maximum number of steps.
    :param accumulate: (int) number of batches to accumulate.
    :param batch_size: (int) batch size.
    :param image_encoder: (str) image encoder name.
    :param text_encoder: (str) text encoder name.
    :param image_dim_out: (int) image width.
    :param text_dim_out: (int) text width.
    :param optimizer: (torch.nn.Optimizer) torch optimizer.
    :return:
    """

    text = f"""
    CLIP TRAINING
    ____________________________________________
    Device:         {torch.cuda.get_device_name() if device == torch.device("cuda:0") else "CPU"}
    Epochs:         {epochs}
    Max Steps:      {max_steps}
    Use Checkpoint: {use_checkpoint}
    Accumulate:     {accumulate}
    Batch size:     {batch_size}
    Image Encoder:  ViT{image_encoder} (dim = {image_dim_out})
    Text Encoder:   {text_encoder}  (dim = {text_dim_out})
    Optimizer:      {optimizer}
    _____________________________________________
    """
    print(text)