import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse

from src.trainer import training
from src.data.data_loader import ImageQueryDataset

from src.utils import CLIPLoss
from src.models.CLIP_model import CLIPModule

from src.models.computer_vision.backbones.vit import ViTat224, ViTat336
from src.models.computer_vision.backbones.resnet34 import RN34at224, RN34at336
from src.models.natural_language_processing.nlp_backbones import TransformerB, TransformerL


parser = argparse.ArgumentParser(
    prog='CLIP Trainner.',
    description='CLIP training cycle with evaluation.',
    epilog='The training cycle for CLIP has two options, fine-tuning and training-loop. Fine-tuning occurs'\
           'for one single epoch on the specified model. training-loop runs the whole loop and requires all parameters'\
           'to be set'
)

# Trainer mode
parser.add_argument('-fine_tuning', type=bool, default=True, help='Perform Fine tuning over one epoch. Requires arg model different from default:None.')
parser.add_argument('-device', type=str, default="cpu", help="Set device to use: gpu or cpu.")

# CLIP Hyper-parameters
parser.add_argument('-image_encoder', type=str, default=None, help="Image encoder backbone. One of ViT@224, ViT@336, RN@224, or RN@336.")
parser.add_argument('-text_encoder', type=str, default=None, help="Text encoder backbone. One of Base or Large.")
parser.add_argument('-max_temperature', type=float, default=100.0, help="Maximum temperature for CLIP loss.")
parser.add_argument('-batch_size', type=int, default=8, help="Batch size. Is the same as the multimodal embedding dimension.")
parser.add_argument('-epochs', type=int, default=32, help="Epochs for training. (ignored in fine-tuning).")
parser.add_argument('-vocab_size', type=int, default=20000, help="Vocabulary size from trained tokenizer.")
parser.add_argument('-decay', type=float, default=0.2, help="Weight decay.")
parser.add_argument('-beta_1', type=float, default=0.9, help="Adam optimizer beta_1.")
parser.add_argument('-beta_2', type=float, default=0.98, help="Adam optimizer beta_2. Recommended 0.98 for ViT and 0.999 for ResNet34.")
parser.add_argument('-epsilon', type=float, default=1e-6, help="Adam optimizer epsilon. Recommended 1e-6 for ViT and 1e-8 for ResNet34.")
parser.add_argument('-lr', type=float, default=2e-5, help="Learning rate.")
parser.add_argument('-text_dim_out', type=int, default=512, help="Text encoder output dimension.")
parser.add_argument('-image_dim_out', type=int, default=768, help="Image encoder output dimension.")
parser.add_argument('-embedding_dim', type=int, default=512, help="Embedding dimension CLIP.")


args = parser.parse_args()

if __name__ == "__main__":
    if args.fine_tuning:
        epochs = 1
    else:
        epochs = args.epochs

    # Get multimodal embedding dim which is equal to the batch size
    multimodal_embedding_dim = args.batch_size

    # Get device
    device = torch.device('cuda:0') if args.device=="gpu" else torch.device('cpu')

    # Pick Image Encoder model
    assert args.image_encoder in ['ViT@224', 'ViT@336', 'RN@224', 'RN@336']

    image_model = None
    if args.image_encoder == "ViT@224":
        image_model = ViTat224(dim_out=args.image_dim_out)
    if args.image_encoder == "ViT@336":
        image_model = ViTat336(dim_out=args.image_dim_out)
    if args.image_encoder == "RN@224":
        image_model = RN34at224(dim_out=args.image_dim_out)
    if args.image_encoder == "RN@336":
        image_model = RN34at336(dim_out=args.image_dim_out)

    # Pick Text Encoder model
    assert args.image_encoder in ['ViT@224', 'ViT@336', 'RN@224', 'RN@336']

    text_model = None
    if args.text_encoder == "Base":
        text_model = TransformerB(dim_out=args.text_dim_out, vocab_size=args.vocab_size, max_length=76)
    if args.text_encoder == "Large":
        text_model = TransformerL(dim_out=args.text_dim_out, vocab_size=args.vocab_size, max_length=76)

    # Call CLIP core model
    clip_model = CLIPModule(image_encoder=image_model, text_encoder=text_model, dim_img=args.image_dim_out, dim_text=args.text_dim_out, embedding_dim=args.embedding_dim, temperature=0.07)

    # Set CLIP Loss function
    loss_func = CLIPLoss(logits_length=multimodal_embedding_dim)

    # Set Adam Optimizer
    optimizer = torch.optim.Adam(clip_model.parameters(), lr=args.lr, eps=args.epsilon, betas=(args.beta_1, args.beta_2))

    # Load training dataset
    training_dataset = ImageQueryDataset(data_dir="src/data/image_gen/WQ-dataset", filename="image-queries-cap-at-10000.json")
    dataloader = DataLoader(training_dataset, batch_size=multimodal_embedding_dim, shuffle=True, num_workers=4)

    # Training cycle
    training(training_dataset=training_dataset, clip_model=clip_model, loss_function=loss_func, optimizer=optimizer, epochs=epochs, device=device)



