from tqdm import tqdm


def training(training_dataset, clip_model, loss_function, optimizer, epochs, device):
    for epoch in range(0, epochs):
        pbar = tqdm(total=len(training_dataset))
        for images, queries in training_dataset:
            images.to(device)
            queries.to(device)

            # Extract feature representations
            logits = clip_model(images, queries)

            # Initialize Gradient
            optimizer.zero_grad()

            # Compute Loss
            loss = loss_function(logits)

            # Backpropagation
            loss.backward()

            # Optimization
            optimizer.step()

            pbar.set_description(f"Epoch:{epoch}. CURR LOSS:{loss.item()}")
            pbar.update(1)