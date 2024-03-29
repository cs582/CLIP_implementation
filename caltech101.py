import os
import tqdm
import argparse
import numpy as np
import torchvision.transforms as T

from tokenizers import Tokenizer
from src.utils import load_from_checkpoint

from src.models.CLIP_model import CLIPModule
from src.models.computer_vision.backbones.vit import ViTBaseOver16at112, ViTBaseOver32at224, ViTSmallOver16at112, ViTMicroOver14at112
from src.models.natural_language_processing.nlp_backbones import GPTSmall, GPTBase, GPTLarge

import torch
import torchvision

parser = argparse.ArgumentParser(
    prog='Caltech101 evaluation.',
    description='Caltech101 evaluation of CLIP.'
)

parser.add_argument('-clip', type=str, default="B224px", help="Choose CLIP Model")
parser.add_argument('-epoch', type=int, default=3, help="Choose Training Stage.")

args = parser.parse_args()

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

def load_clip(clip_model, epoch):
    """
    Load CLIP model.
    """
    checkpointsdir = "src/models/checkpoints"

    if clip_model == "B224px":
        clip = load_clip_backbone(image_encoder="B/32@224", text_encoder="B", device=torch.device('cpu'))
        if epoch == -1:
            return clip, 224
        if epoch == 0:
            _, loss_hist = load_from_checkpoint(os.path.join(checkpointsdir, "CLIP_epoch_0_2023-06-26_10:18:36"), clip)
        if epoch == 1:
            _, loss_hist = load_from_checkpoint(os.path.join(checkpointsdir, "CLIP_epoch_1_2023-06-28_06:12:08"), clip)
        if epoch == 2:
            _, loss_hist = load_from_checkpoint(os.path.join(checkpointsdir, "CLIP_epoch_2_2023-06-30_01:36:39"), clip)
        if epoch == 3:
            _, loss_hist = load_from_checkpoint(os.path.join(checkpointsdir, "CLIP_epoch_3_2023-07-01_21:04:05"), clip)
        return clip, 224

    if clip_model == "B112px":
        clip = load_clip_backbone(image_encoder="B/16@112", text_encoder="B", device=torch.device('cpu'))
        if epoch == -1:
            return clip, 112
        if epoch == 0:
            _, loss_hist = load_from_checkpoint(os.path.join(checkpointsdir, "CLIP_epoch_0_2023-07-06_08:11:02"), clip)
        if epoch == 1:
            _, loss_hist = load_from_checkpoint(os.path.join(checkpointsdir, "CLIP_epoch_1_2023-07-08_04:22:18"), clip)
        if epoch == 2:
            _, loss_hist = load_from_checkpoint(os.path.join(checkpointsdir, "CLIP_epoch_2_2023-07-09_23:50:00"), clip)
        if epoch == 3:
            _, loss_hist = load_from_checkpoint(os.path.join(checkpointsdir, "CLIP_epoch_3_2023-07-13_09:59:29"), clip)
        return clip, 112


if __name__=="__main__":

    transform = T.ToTensor()

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

    image_path = "data/caltech101"
    tokenizer_file = "src/data/nlp/tokenizers/CLIP-bpe.tokenizer.json"

    device = f'cuda:1'
    use_checkpoint = False

    model_size = args.clip
    epoch = args.epoch

    vocab_size = 20000
    clip_embedding_dim = 512
    max_length = 32

    text_dim_out = 512
    image_dim_out = 768

    print("Started...")
    testset = torchvision.datasets.ImageFolder(root='./data/caltech101/101_ObjectCategories',
                                               transform=T.ToTensor(),
                                               target_transform=T.Lambda(lambda y: torch.tensor(y)))

    classes = testset.classes

    print("Loading Model...")
    clip, img_res = load_clip(model_size, epoch)

    tokenizer = Tokenizer.from_file(tokenizer_file)

    encode_text = lambda x, k : tokenize(tokenizer, x.format(k), 32)

    print("Calculating Zero-shoot Weights...")
    zero_shot_weights = torch.zeros(len(classes), clip_embedding_dim)
    for i, key in tqdm.tqdm(enumerate(classes), total=len(classes)):
        class_tokens = torch.from_numpy( np.array( [ encode_text(x, key) for x in templates ] ) )
        text_encoding = clip.txt_encoder(class_tokens)
        zero_shot_weights[i, :] = clip.txt_encoder(class_tokens).mean(dim=0)

    correct = 0


    def treat_image(img):

        _, h, w = img.shape
        factor = img_res / min(w, h)

        new_width = int(w * factor) + 1
        new_height = int(h * factor) + 1

        img = T.Resize((new_height, new_width), antialias=True)(img)
        img = T.RandomCrop((img_res, img_res))(img)

        return img

    print("Evaluating images...")
    n_batches = len(testset) // batch_size
    for i in tqdm.tqdm(range(n_batches), total=n_batches):
        p0, p1 = i*batch_size, (i+1)*batch_size

        images = torch.stack([treat_image(testset[k][0]) for k in range(p0, p1)])
        y_true = torch.stack([testset[k][1] for k in range(p0, p1)])

        image_encoding = clip.img_encoder(images)

        y_hat = (image_encoding @ zero_shot_weights.t()).argmax(dim=-1)
        correct += (y_hat == y_true).sum().item()

    print("Zero-Shoot Classification on Caltech101", (correct / len(testset)) * 100.0)