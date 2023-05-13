import boto3

from tqdm import tqdm
from src.utils import load_from_checkpoint


s3 = boto3.client('s3')

# Models directory
models_dir = "src/models/checkpoints"


import torch

def eval(eval_dataset, clip_model, loss_function, device, load_from_given_checkpoint=None):
    """
    Evaluation loop.
    :param eval_dataset: (Dataloader) evaluation data.
    :param clip_model: (torch.nn.Module) CLIP model.
    :param loss_function: (torch.nn.Module) CLIP loss function.
    :param device: (torch.device) Torch device.
    :param load_from_given_checkpoint: (str) Path to the checkpoint to load.
    :return: top1_accuracy, top5_accuracy
    """
    _, _ = load_from_checkpoint(load_from_given_checkpoint, clip_model)

    loss_eval = []
    top1_correct = 0
    top5_correct = 0
    total_samples = 0

    pbar = tqdm(total=len(eval_dataset), desc="Eval. Loop.")
    for idx, (images, queries) in enumerate(eval_dataset):
        images, queries = images.to(device), queries.to(device)

        # Extract feature representations
        logits_text, logits_images = clip_model(queries)

        # Compute Loss
        loss = loss_function(logits_images, logits_text)
        loss_eval.append(loss.item())

        # Calculate Top-1 and Top-5 accuracies
        predictions = torch.topk(logits_text, k=5, dim=1)[1]
        labels = torch.arange(logits_text.size(0)).to(device)
        top1_correct += torch.sum(predictions[:, 0] == labels).item()
        top5_correct += torch.sum(predictions == labels.view(-1, 1)).item()
        total_samples += labels.size(0)

        pbar.update(1)

    pbar.close()

    top1_accuracy = top1_correct / total_samples
    top5_accuracy = top5_correct / total_samples

    print(f"Top-1 Accuracy: {top1_accuracy:.4f}, Top-5 Accuracy: {top5_accuracy:.4f}")

    return top1_accuracy, top5_accuracy