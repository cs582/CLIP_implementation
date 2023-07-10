import os
import tqdm
import torchvision.transforms as T

from torchvision.datasets import CocoCaptions
from tokenizers import Tokenizer
from src.utils import load_from_checkpoint

from src.models.CLIP_model import CLIPModule
from src.models.computer_vision.backbones.vit import ViTBaseOver16at112, ViTBaseOver32at224, ViTSmallOver16at112, ViTMicroOver14at112
from src.models.natural_language_processing.nlp_backbones import GPTSmall, GPTBase, GPTLarge

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


device = f'cuda:1'

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
        clip = load_clip_backbone(image_encoder="B/32@224", text_encoder="B", device=torch.device(device))
        clip, _, loss_hist = load_from_checkpoint(os.path.join(checkpointsdir, "CLIP_epoch_3_2023-07-01_21:04:05"), clip)
        return clip, 224

    if clip_model == "B112px":
        clip = load_clip_backbone(image_encoder="B/16@112", text_encoder="B", device=torch.device(device))
        clip, _, loss_hist = load_from_checkpoint(os.path.join(checkpointsdir, "CLIP_epoch_2_2023-07-09_23:50:00"), clip)
        return clip, 112


def accuracy(output, target, topk=(1,)):
    """
    Code source: github.com/openai/CLIP
    """
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


if __name__=="__main__":

    transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    batch_size = 64

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

    data_dir = "data/coco/annotations"
    tokenizer_file = "src/data/nlp/tokenizers/CLIP-bpe.tokenizer.json"

    use_checkpoint = False

    vocab_size = 20000
    clip_embedding_dim = 512
    max_length = 32

    text_dim_out = 512
    image_dim_out = 768

    # Define the transformations to be applied to the images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # Create an instance of the ImageNet dataset
    dataset = CocoCaptions(root=data_dir, annFile=os.path.join(data_dir, 'captions_val2017.json'), transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    print("Loading Model...")
    model_size = "B224px"
    clip, img_res = load_clip(model_size)

    tokenizer = Tokenizer.from_file(tokenizer_file)

    encode_text = lambda x : tokenize(tokenizer, x, 32)

    correct = 0

    print("Evaluating images...")
    for images, captions in tqdm.tqdm(dataloader, total=len(dataloader)):
        images = images.to(device)
        tokens = torch.Tensor([encode_text(x[0]) for x in captions]).to(device, dtype=torch.int)

        logits, _ = clip(images, tokens)
        y_hat = logits.argmax(dim=-1)

        correct += ( y_hat == torch.Tensor(np.arange(len(y_hat))).to(device) ).sum().item()

    print("COCO Text-Image Connection Performance against 8 images")
    top1 = (correct / len(dataset)) * 100

    print(f"Top-1 accuracy: {top1:.2f}")