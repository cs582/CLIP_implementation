import os
import torch
from torch.optim.lr_scheduler import LambdaLR
from datetime import datetime as dt


def warmup_scheduler(optimizer, warmup_steps, lr_max):
    lr_lambda = lambda step: step * lr_max / warmup_steps if step < warmup_steps else 1.0
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(model, optimizer, epoch, scheduler, history, models_dir):
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


def load_from_checkpoint(model_filepath, model, scheduler, optimizer):

    checkpoint = torch.load(model_filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epoch']
    loss_history = checkpoint['loss_history']

    print(f"CLIP successfully loaded from {model_filepath}")

    return epoch, loss_history


def training_info_log_message(device, epochs, batch_size, image_encoder, text_encoder, image_dim_out, text_dim_out, optimizer):

    text = f"""
    CLIP TRAINING
    ____________________________________________
    Device:         {torch.cuda.get_device_name() if device == torch.device("cuda:0") else "CPU"}
    Epochs:         {epochs}
    Batch size:     {batch_size}
    Image Encoder:  ViT{image_encoder} (dim = {image_dim_out})
    Text Encoder:   {text_encoder}  (dim = {text_dim_out})
    Optimizer:      {optimizer}
    _____________________________________________
    """

    print(text)