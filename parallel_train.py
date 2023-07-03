import torch
import argparse

from src.parallel_trainer import parallel_training
from torch.utils.data import DataLoader
from src.data.data_loader import ImageQueryDataset

from src.models.CLIP_Loss import CLIPLoss
from src.models.CLIP_model import CLIPModule

from src.models.computer_vision.backbones.vit import ViTBaseOver16at112, ViTSmallOver16at112, ViTMicroOver14at112
from src.models.natural_language_processing.nlp_backbones import GPTSmall, GPTBase


from src.utils import training_info_log_message


parser = argparse.ArgumentParser(
    prog='CLIP Trainner.',
    description='CLIP training cycle with evaluation.',
    epilog='The training cycle for CLIP has two options, fine-tuning and training-loop. Fine-tuning occurs' \
           'for one single epoch on the specified model. training-loop runs the whole loop and requires all parameters' \
           'to be set'
)

# Trainer mode
parser.add_argument('-warmup', type=int, default=2000, help="Warmup steps.")
parser.add_argument('-use_checkpoint', type=bool, default=False, help="Use checkpointing for training.")

# CLIP Hyper-parameters
parser.add_argument('-max_temperature', type=float, default=100.0, help="Maximum temperature for CLIP loss.")
parser.add_argument('-batch_size', type=int, default=128, help="Batch size. Is the same as the multimodal embedding dimension.")
parser.add_argument('-epochs', type=int, default=4, help="Epochs for training. (ignored in fine-tuning).")
parser.add_argument('-vocab_size', type=int, default=20000, help="Vocabulary size from trained tokenizer.")
parser.add_argument('-max_length', type=int, default=32, help="Max length of the token encoding.")
parser.add_argument('-decay', type=float, default=0.2, help="Weight decay.")
parser.add_argument('-beta_1', type=float, default=0.9, help="Adam optimizer beta_1.")
parser.add_argument('-beta_2', type=float, default=0.98, help="Adam optimizer beta_2. Recommended 0.98 for ViT.")
parser.add_argument('-epsilon', type=float, default=1e-6, help="Adam optimizer epsilon. Recommended 1e-6 for ViT.")
parser.add_argument('-lr', type=float, default=4e-5, help="Learning rate.")
parser.add_argument('-text_dim_out', type=int, default=512, help="Text encoder output dimension.")
parser.add_argument('-image_dim_out', type=int, default=768, help="Image encoder output dimension.")
parser.add_argument('-clip_embedding_dim', type=int, default=512, help="Embedding dimension CLIP.")

args = parser.parse_args()

dataset_file = "src/data/image_gen/WQ-dataset/WQI_local.csv"
image_path = "/data/carlos/images"
tokenizer_file = "src/data/nlp/tokenizers/CLIP-bpe.tokenizer.json"


if __name__ == "__main__":
    # Trainer mode
    warmup_steps = args.warmup
    use_checkpoint = args.use_checkpoint

    # CLIP Hyper-parameters
    max_temperature = args.max_temperature
    batch_size = args.batch_size
    epochs = args.epochs
    vocab_size = args.vocab_size
    max_length = args.max_length

    # Optimizer parameters
    decay = args.decay
    beta_1 = args.beta_1
    beta_2 = args.beta_2
    epsilon = args.epsilon
    lr = args.lr

    # Model size parameters
    text_dim_out = args.text_dim_out
    image_dim_out = args.image_dim_out
    clip_embedding_dim = args.clip_embedding_dim

    # Get multimodal embedding dim which is equal to the batch size
    multimodal_embedding_dim = args.batch_size

    # Image Enocders
    image_encoder_names = ["B/16@112", "S/16@112", "M/14@112"]
    image_model_0 = ViTBaseOver16at112(dim_out=image_dim_out).to('cuda:0')
    image_model_1 = ViTSmallOver16at112(dim_out=image_dim_out).to('cuda:1')
    image_model_2 = ViTMicroOver14at112(dim_out=image_dim_out).to('cuda:2')
    image_resolution = 112

    # Text Encoders
    text_encoder_names = ['GPT Base', 'GPT Small', 'GPT Small']
    text_model_0 = GPTBase(dim_out=text_dim_out, vocab_size=vocab_size, max_length=max_length, use_checkpoint=use_checkpoint, device='cuda:0').to('cuda:0')
    text_model_1 = GPTSmall(dim_out=text_dim_out, vocab_size=vocab_size, max_length=max_length, use_checkpoint=use_checkpoint, device='cuda:1').to('cuda:1')
    text_model_2 = GPTSmall(dim_out=text_dim_out, vocab_size=vocab_size, max_length=max_length, use_checkpoint=use_checkpoint, device='cuda:2').to('cuda:2')

    # Load training dataset
    training_dataset = ImageQueryDataset(dataset_file=dataset_file, image_path=image_path, tokenizer_file=tokenizer_file, max_length=max_length, img_res=image_resolution)
    dataloader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # Calculate max-steps
    max_steps = len(dataloader) * epochs

    # Set CLIP Loss function
    losses = [CLIPLoss(logits_length=multimodal_embedding_dim).to(f'cuda:{i}') for i in range(3)]

    # Call CLIP core model
    clip_model_0 = CLIPModule(image_encoder=image_model_0, text_encoder=text_model_0, dim_img=image_dim_out, dim_text=text_dim_out, embedding_dim=clip_embedding_dim, temperature=0.07).to('cuda:0')
    clip_model_1 = CLIPModule(image_encoder=image_model_1, text_encoder=text_model_1, dim_img=image_dim_out, dim_text=text_dim_out, embedding_dim=clip_embedding_dim, temperature=0.07).to('cuda:1')
    clip_model_2 = CLIPModule(image_encoder=image_model_2, text_encoder=text_model_2, dim_img=image_dim_out, dim_text=text_dim_out, embedding_dim=clip_embedding_dim, temperature=0.07).to('cuda:2')
    clip_models = (clip_model_0, clip_model_1, clip_model_2)

    # Set Adam Optimizer
    optimizer_0 = torch.optim.AdamW(clip_model_0.parameters(), lr=lr, eps=epsilon, betas=(beta_1, beta_2), weight_decay=decay)
    optimizer_1 = torch.optim.AdamW(clip_model_1.parameters(), lr=lr, eps=epsilon, betas=(beta_1, beta_2), weight_decay=decay)
    optimizer_2 = torch.optim.AdamW(clip_model_2.parameters(), lr=lr, eps=epsilon, betas=(beta_1, beta_2), weight_decay=decay)
    optimizers = [optimizer_0, optimizer_1, optimizer_2]

    # Print training information
    for i, (image_encoder_name, text_encoder_name, optimizer) in enumerate(zip(image_encoder_names, text_encoder_names, optimizers)):
        training_info_log_message(device=f'cuda:{i}', use_checkpoint=use_checkpoint, vocab_size=vocab_size, epochs=epochs, max_steps=max_steps, batch_size=batch_size, image_encoder=image_encoder_name, text_encoder=text_encoder_name, image_dim_out=image_dim_out, text_dim_out=text_dim_out, optimizer=optimizer)

    # Training cycle
    parallel_training(training_dataset=dataloader, clip_models=clip_models, losses=losses, optimizers=optimizers, epochs=epochs, lr_max=lr, warmup_steps=warmup_steps, max_steps=max_steps, model_names=image_encoder_names)
