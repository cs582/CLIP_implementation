import os
import tqdm
import torchvision.transforms as T

from torchvision.datasets import ImageNet
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

    image_path = "data/imagenet"
    class_map_path = "data/imagenet/imagenet_labels.txt"
    tokenizer_file = "src/data/nlp/tokenizers/CLIP-bpe.tokenizer.json"

    classes = [None] * 1000
    with open(class_map_path, 'r') as f:
        for i, line in enumerate(f):
            classes[i] = line.split(",")[0].strip()

    device = f'cuda:1'
    use_checkpoint = False

    vocab_size = 20000
    clip_embedding_dim = 512
    max_length = 32

    text_dim_out = 512
    image_dim_out = 768

    print("Started...")
    os.makedirs(image_path, exist_ok=True)

    with open(class_map_path, 'r') as label_file:
        for class_id, line in enumerate(label_file):
            class_name = line.split(",")[0].strip()
            class_dir = os.path.join(image_path, str(class_id))
            os.makedirs(class_dir, exist_ok=True)

    # Define the transformations to be applied to the images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # Create an instance of the ImageNet dataset
    dataset = ImageNet(root=image_path, split='val', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

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

    top1 = 0
    top5 = 0
    top10 = 0

    print("Evaluating images...")
    for images, y_true in tqdm.tqdm(dataloader, total=len(dataloader)):
        _, h, w = images[0].shape
        factor = img_res / min(w, h)

        new_width = int(w * factor)
        new_height = int(h * factor)

        images = T.Resize((new_height, new_width), antialias=True)(images)
        images = T.RandomCrop((img_res, img_res))(images)

        image_encoding = clip.img_encoder(images)

        # ---
        y_hat = 100. * image_encoding @ zero_shot_weights.t()
        acc1, acc5, acc10 = accuracy(y_hat, y_true, topk=(1, 5, 10))
        top1 += acc1
        top5 += acc5
        top10 += acc10
        # ---- Source github.com/openai/CLIP

    print("Zero-Shoot Classification on ImageNet Val")
    top1 = (top1 / len(dataset)) * 100
    top5 = (top5 / len(dataset)) * 100
    top10 = (top10 / len(dataset)) * 100

    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")
    print(f"Top-10 accuracy: {top10:.2f}")