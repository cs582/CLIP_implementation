import torch
import argparse

from src.trainer import training
from torch.utils.data import DataLoader
from src.data.data_loader import ImageQueryDataset

from src.models.CLIP_Loss import CLIPLoss
from src.models.CLIP_model import CLIPModule

from src.models.computer_vision.backbones.vit import ViTBaseOver16at112, ViTBaseOver32at224, ViTSmallOver16at112, ViTMicroOver14at112
from src.models.natural_language_processing.nlp_backbones import GPTSmall, GPTBase, GPTLarge

from src.utils import training_info_log_message, warmup_scheduler


parser = argparse.ArgumentParser(
    prog='CLIP Trainner.',
    description='CLIP training cycle with evaluation.',
    epilog='The training cycle for CLIP has two options, fine-tuning and training-loop. Fine-tuning occurs'\
           'for one single epoch on the specified model. training-loop runs the whole loop and requires all parameters'\
           'to be set'
)

# Trainer mode
parser.add_argument('-device', type=str, default="cpu", help="Set device to use: gpu or cpu.")
parser.add_argument('-load_last_checkpoint', type=bool, default=False, help="Load model from last checkpoint and restart training from there.")
parser.add_argument('-warmup', type=int, default=2000, help="Warmup steps.")
parser.add_argument('-use_checkpoint', type=bool, default=False, help="Use checkpointing for training.")

# CLIP Hyper-parameters
parser.add_argument('-image_encoder', type=str, default=None, help="Image encoder backbone. One of (ViT) @112, @224, or @336.")
parser.add_argument('-text_encoder', type=str, default=None, help="Text encoder backbone. One of S (Small), B (Base), or L (Large).")
parser.add_argument('-max_temperature', type=float, default=100.0, help="Maximum temperature for CLIP loss.")
parser.add_argument('-batch_size', type=int, default=128, help="Batch size. Is the same as the multimodal embedding dimension.")
parser.add_argument('-epochs', type=int, default=5, help="Epochs for training. (ignored in fine-tuning).")
parser.add_argument('-vocab_size', type=int, default=20000, help="Vocabulary size from trained tokenizer.")
parser.add_argument('-max_length', type=int, default=32, help="Max length of the token encoding.")
parser.add_argument('-decay', type=float, default=0.2, help="Weight decay.")
parser.add_argument('-beta_1', type=float, default=0.9, help="Adam optimizer beta_1.")
parser.add_argument('-beta_2', type=float, default=0.98, help="Adam optimizer beta_2. Recommended 0.98 for ViT.")
parser.add_argument('-epsilon', type=float, default=1e-6, help="Adam optimizer epsilon. Recommended 1e-6 for ViT.")
parser.add_argument('-lr', type=float, default=5e-4, help="Learning rate.")
parser.add_argument('-text_dim_out', type=int, default=512, help="Text encoder output dimension.")
parser.add_argument('-image_dim_out', type=int, default=768, help="Image encoder output dimension.")
parser.add_argument('-embedding_dim', type=int, default=512, help="Embedding dimension CLIP.")

args = parser.parse_args()

dataset_file = "src/data/image_gen/WQ-dataset/WKIT_local.csv"
image_path = "/data/carlos/images"
tokenizer_file = "src/data/nlp/tokenizers/CLIP-bpe.tokenizer.json"


if __name__ == "__main__":
    # CLIP Hyper-parameters
    image_encoder = args.image_encoder
    text_encoder = args.text_encoder
    max_temperature = args.max_temperature
    vocab_size = args.vocab_size
    clip_embedding_dim = args.embedding_dim

    # Training Hyper-parameters
    batch_size = args.batch_size
    max_length = args.max_length
    epochs = args.epochs
    warmup = args.warmup
    decay = args.decay
    beta_1 = args.beta_1
    beta_2 = args.beta_2
    epsilon = args.epsilon
    lr = args.lr

    # Text-encoder Hyper-parameters
    text_dim_out = args.text_dim_out

    # Image-encoder Hyper-parameters
    image_dim_out = args.image_dim_out

    # Other Hyper-parameters
    load_last_checkpoint = args.load_last_checkpoint
    use_checkpoint = args.use_checkpoint

    # Get device
    device = torch.device('cuda:0') if args.device=="gpu" else torch.device('cpu')

    # Pick Image Encoder model
    assert image_encoder in ['B/32@224', 'B/16@112', 'S/8@112', 'M/14@112']

    image_model = None
    image_resolution = None
    if image_encoder == "B/32@224":
        image_model = ViTBaseOver32at224(dim_out=image_dim_out).to(device)
        image_resolution = 224
    if image_encoder == "B/16@112":
        image_model = ViTBaseOver16at112(dim_out=image_dim_out).to(device)
        image_resolution = 112
    if image_encoder == "S/16@112":
        image_model = ViTSmallOver16at112(dim_out=image_dim_out).to(device)
        image_resolution = 112
    if image_encoder == "M/14@112":
        image_model = ViTMicroOver14at112(dim_out=image_dim_out).to(device)
        image_resolution = 112

    # Pick Text Encoder model
    assert text_encoder in ['S', 'B', 'L']

    text_model = None
    if text_encoder == "S":
        text_model = GPTSmall(dim_out=text_dim_out, vocab_size=vocab_size, max_length=max_length, use_checkpoint=use_checkpoint, device=device).to(device)
    if text_encoder == "B":
        text_model = GPTBase(dim_out=text_dim_out, vocab_size=vocab_size, max_length=max_length, use_checkpoint=use_checkpoint, device=device).to(device)
    if text_encoder == "L":
        text_model = GPTLarge(dim_out=text_dim_out, vocab_size=vocab_size, max_length=max_length, use_checkpoint=use_checkpoint, device=device).to(device)

    # Load training dataset
    training_dataset = ImageQueryDataset(dataset_file=dataset_file, image_path=image_path, tokenizer_file=tokenizer_file, max_length=max_length, img_res=image_resolution)
    dataloader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # Calculate max-steps
    max_steps = len(dataloader) * epochs

    # Set CLIP Loss function
    loss_func = CLIPLoss(logits_length=batch_size).to(device)

    # Call CLIP core model
    clip_model = CLIPModule(image_encoder=image_model, text_encoder=text_model, dim_img=image_dim_out, dim_text=text_dim_out, embedding_dim=clip_embedding_dim, temperature=0.07).to(device)

    # Set Adam Optimizer
    optimizer = torch.optim.AdamW(clip_model.parameters(), lr=lr, eps=epsilon, betas=(beta_1, beta_2), weight_decay=decay)

    # Warm-up scheduler(optimizer, warmup_steps, warmup_start, lr_max, max_steps)
    scheduler = warmup_scheduler(optimizer, warmup_steps=warmup, warmup_start=0.0, lr_max=lr, max_steps=max_steps)

    # Print training information
    training_info_log_message(device=device, use_checkpoint=use_checkpoint, vocab_size=vocab_size, epochs=epochs, max_steps=max_steps, batch_size=batch_size, image_encoder=image_encoder, text_encoder=text_encoder, image_dim_out=image_dim_out, text_dim_out=text_dim_out, optimizer=optimizer)

    # Training cycle
    training(training_dataset=dataloader, clip_model=clip_model, loss_function=loss_func, optimizer=optimizer, scheduler=scheduler, epochs=epochs, device=device, model_name=image_encoder, load_last_checkpoint=load_last_checkpoint)
