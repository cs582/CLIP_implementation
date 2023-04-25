import torch


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