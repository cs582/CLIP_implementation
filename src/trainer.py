from tqdm import tqdm
import numpy as np


def training(training_dataset, clip_model, loss_function, optimizer, epochs, device):
    for epoch in range(0, epochs):
        pbar = tqdm(total=len(training_dataset))
        for images, queries in training_dataset:
            images, queries = images.to(device), queries.to(device)

            # Extract feature representations
            logits_images, logits_text = clip_model(images, queries)

            # Initialize Gradient
            optimizer.zero_grad()

            # Compute Loss
            loss = loss_function(logits_images, logits_text)

            # Backpropagation
            loss.backward()

            # Optimization
            optimizer.step()

            pbar.set_description(f"Epoch:{epoch}. CURR LOSS:{np.round(loss.item(),3)}")
            pbar.update(1)