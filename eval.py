import torch
import argparse


from src.eval_loop import eval
from torch.utils.data import DataLoader
from src.data.data_loader import ImageQueryDataset

from src.models.CLIP_Loss import CLIPLoss
from src.models.CLIP_model import CLIPModule

from src.models.computer_vision.backbones.vit import ViTBaseOver16at112, ViTBaseOver32at224, ViTSmallOver16at112, ViTMicroOver14at112
from src.models.natural_language_processing.nlp_backbones import GPTBase, GPTLarge

from src.utils import training_info_log_message


parser = argparse.ArgumentParser(
    prog='CLIP Evaluator.',
    description='CLIP evaluation loop.',
    epilog='The CLIP evaluation loop computes the loss and accuracy across the given dataset.'
)

# Trainer mode
parser.add_argument('-device', type=str, default="cpu", help="Set device to use: gpu or cpu.")
parser.add_argument('-checkpoint', type=bool, default=False, help="Load model from last checkpoint and restart training from there.")
parser.add_argument('-dataset', type=str, default="imagenet", help="Specify the dataset to load data from.")

# CLIP Hyper-parameters
parser.add_argument('-image_encoder', type=str, default=None, help="Image encoder backbone. One of (ViT) @112, @224, or @336.")
parser.add_argument('-text_encoder', type=str, default=None, help="Text encoder backbone. One of S (Small), B (Base), or L (Large).")
parser.add_argument('-batch_size', type=int, default=128, help="Batch size. Is the same as the multimodal embedding dimension.")
parser.add_argument('-vocab_size', type=int, default=43001, help="Vocabulary size from trained tokenizer.")
parser.add_argument('-max_length', type=int, default=34, help="Max length of the token encoding.")
parser.add_argument('-text_dim_out', type=int, default=512, help="Text encoder output dimension.")
parser.add_argument('-image_dim_out', type=int, default=768, help="Image encoder output dimension.")
parser.add_argument('-embedding_dim', type=int, default=512, help="Embedding dimension CLIP.")

args = parser.parse_args()

dataset_file = None
image_path = None
tokenizer_file = "src/data/nlp/tokenizers/CLIP-bpe.tokenizer.json"

if __name__ == "__main__":

    if args.dataset == "imagenet":
        dataset_file = "data/imagenet/imagenet.csv"
        image_path = "data/imagenet/images"
    if args.dataset == "cryptopunks":
        dataset_file = "data/cryptopunks/cryptopunks.csv"
        image_path = "data/cryptopunks/imgs/imgs"
    if args.dataset == "cifar10":
        dataset_file = "data/cifar10/cifar10.csv"
        image_path = "data/cifar10/images"

    # Get multimodal embedding dim which is equal to the batch size
    multimodal_embedding_dim = args.batch_size

    # Get device
    device = torch.device('cuda:0') if args.device=="gpu" else torch.device('cpu')

    # Pick Image Encoder model
    assert args.image_encoder in ['B/32@224', 'B/16@112', 'S/8@112', 'S/16@112']

    # Enable cublasLt for mixed-precision training
    torch.backends.cudnn.enabled = True

    image_model = None
    image_resolution = None
    if args.image_encoder == "B/32@224":
        image_model = ViTBaseOver32at224(dim_out=args.image_dim_out).to(device)
        image_resolution = 224
    if args.image_encoder == "B/16@112":
        image_model = ViTBaseOver16at112(dim_out=args.image_dim_out).to(device)
        image_resolution = 112
    if args.image_encoder == "S/16@112":
        image_model = ViTSmallOver16at112(dim_out=args.image_dim_out).to(device)
        image_resolution = 112
    if args.image_encoder == "M/14@112":
        image_model = ViTMicroOver14at112(dim_out=args.image_dim_out).to(device)
        image_resolution = 112

    # Pick Text Encoder model
    assert args.text_encoder in ['B', 'L']

    text_model = None
    if args.text_encoder == "B":
        text_model = GPTBase(dim_out=args.text_dim_out, vocab_size=args.vocab_size, max_length=args.max_length, batch_size=args.batch_size).to(device)
    if args.text_encoder == "L":
        text_model = GPTLarge(dim_out=args.text_dim_out, vocab_size=args.vocab_size, max_length=args.max_length, batch_size=args.batch_size).to(device)

    # Load training dataset
    eval_dataset = ImageQueryDataset(dataset_file, image_path, tokenizer_file, args.max_length, device, start_from=7500000, img_res=image_resolution)
    dataloader = DataLoader(eval_dataset, batch_size=multimodal_embedding_dim, num_workers=10, drop_last=True)

    # Set CLIP Loss function
    loss_func = CLIPLoss(logits_length=multimodal_embedding_dim).to(device)

    # Call CLIP core model
    clip_model = CLIPModule(image_encoder=image_model, text_encoder=text_model, dim_img=args.image_dim_out, dim_text=args.text_dim_out, embedding_dim=args.embedding_dim, temperature=0.07).to(device)

    # Print training information
    training_info_log_message(device, None, args.batch_size, args.image_encoder, args.text_encoder, args.image_dim_out, args.text_dim_out, None)

    # Training cycle
    eval(eval_dataset=dataloader, clip_model=clip_model, loss_function=loss_func, device=device, load_from_given_checkpoint=args.checkpoint)
