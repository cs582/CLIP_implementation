import os
import tqdm
import torchvision.transforms as T

from tokenizers import Tokenizer
from src.utils import load_from_checkpoint

from src.models.CLIP_model import CLIPModule
from src.models.computer_vision.backbones.vit import ViTBaseOver16at112, ViTBaseOver32at224, ViTSmallOver16at112, ViTMicroOver14at112
from src.models.natural_language_processing.nlp_backbones import GPTSmall, GPTBase, GPTLarge

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def load_clip_backbone(image_encoder, text_encoder, device):
    """
    Load clip backbone.
    """

    image_model = None
    if image_encoder == "B/32@224":
        image_model = ViTBaseOver32at224(dim_out=image_dim_out).to(device)
    if image_encoder == "B/16@112":
        image_model = ViTBaseOver16at112(dim_out=image_dim_out).to(device)
    if image_encoder == "S/16@112":
        image_model = ViTSmallOver16at112(dim_out=image_dim_out).to(device)
    if image_encoder == "M/14@112":
        image_model = ViTMicroOver14at112(dim_out=image_dim_out).to(device)

    text_model = None
    if text_encoder == "S":
        text_model = GPTSmall(dim_out=text_dim_out, vocab_size=vocab_size, max_length=max_length, use_checkpoint=use_checkpoint, device=device).to(device)
    if text_encoder == "B":
        text_model = GPTBase(dim_out=text_dim_out, vocab_size=vocab_size, max_length=max_length, use_checkpoint=use_checkpoint, device=device).to(device)
    if text_encoder == "L":
        text_model = GPTLarge(dim_out=text_dim_out, vocab_size=vocab_size, max_length=max_length, use_checkpoint=use_checkpoint, device=device).to(device)

    clip_model = CLIPModule(image_encoder=image_model, text_encoder=text_model, dim_img=image_dim_out, dim_text=text_dim_out, embedding_dim=clip_embedding_dim, temperature=0.07).to(device)

    return clip_model


def tokenize(tokenizer, query, max_length):
    """
    Takes a query and returns the token with the right length.
    """
    # Encode sequence
    encoded_query = tokenizer.encode(query).ids

    # Truncate query if necessary
    encoded_query = encoded_query[:max_length-2]

    # Add end_of_sentence token [EOS]
    encoded_query += [tokenizer.token_to_id('[EOS]')]

    # Add padding to encoded sentence
    encoded_query += [0] * (max_length - len(encoded_query) - 1)

    # Add [SOS] and [EOS] tokens
    encoded_query = [tokenizer.token_to_id('[SOS]')] + encoded_query

    return encoded_query

def load_clip(clip_model):
    """
    Load CLIP model.
    """
    checkpointsdir = "src/models/checkpoints"

    if clip_model == "B224px":
        clip = load_clip_backbone(image_encoder="B/32@224", text_encoder="B", device=torch.device('cpu'))
        clip, _, loss_hist = load_from_checkpoint(os.path.join(checkpointsdir, "CLIP_epoch_3_2023-07-01_21:04:05"), clip)
        return clip, 224

    if clip_model == "B112px":
        clip = load_clip_backbone(image_encoder="B/16@112", text_encoder="B", device=torch.device('cpu'))
        clip, _, loss_hist = load_from_checkpoint(os.path.join(checkpointsdir, "CLIP_epoch_2_2023-07-09_23:50:00"), clip)
        return clip, 112


if __name__=="__main__":

    transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    batch_size = 64

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    templates = [
        'a photo of a {}.',
        'a blurry photo of a {}.',
        'a black and white photo of a {}.',
        'a low contrast photo of a {}.',
        'a high contrast photo of a {}.',
        'a bad photo of a {}.',
        'a good photo of a {}.',
        'a photo of a small {}.',
        'a photo of a big {}.',
        'a photo of the {}.',
        'a blurry photo of the {}.',
        'a black and white photo of the {}.',
        'a low contrast photo of the {}.',
        'a high contrast photo of the {}.',
        'a bad photo of the {}.',
        'a good photo of the {}.',
        'a photo of the small {}.',
        'a photo of the big {}.',
    ]

    image_path = "data/imagenet"
    tokenizer_file = "src/data/nlp/tokenizers/CLIP-bpe.tokenizer.json"

    device = f'cuda:1'
    use_checkpoint = False

    vocab_size = 20000
    clip_embedding_dim = 512
    max_length = 32

    text_dim_out = 512
    image_dim_out = 768

    print("Started...")
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    print("Loading Model...")
    model_size = "B224px"
    clip, img_res = load_clip(model_size)

    tokenizer = Tokenizer.from_file(tokenizer_file)

    encode_text = lambda x, k : tokenize(tokenizer, x.format(k), 32)

    print("Calculating Zero-shoot Weights...")
    zero_shot_weights = torch.zeros(len(classes), clip_embedding_dim)
    for i, key in tqdm.tqdm(enumerate(classes), total=len(classes)):
        class_tokens = torch.from_numpy( np.array( [ encode_text(x, key) for x in templates ] ) )
        text_encoding = clip.txt_encoder(class_tokens)
        zero_shot_weights[i, :] = clip.txt_encoder(class_tokens).mean(dim=0)

    correct = 0

    print("Evaluating images...")
    for images, y_true in tqdm.tqdm(testloader, total=len(testloader)):
        _, h, w = images[0].shape
        factor = img_res / min(w, h)

        new_width = int(w * factor)
        new_height = int(h * factor)

        images = T.Resize((new_height, new_width), antialias=True)(images)
        images = T.RandomCrop((img_res, img_res))(images)

        image_encoding = clip.img_encoder(images)

        y_hat = (image_encoding @ zero_shot_weights.t()).argmax(dim=-1)
        correct += (y_hat == y_true).sum().item()

    print("Zero-Shoot Classification on CIFAR10", (correct / len(testset)) * 100.0)