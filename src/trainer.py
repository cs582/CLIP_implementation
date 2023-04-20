from tqdm import tqdm


def training(training_dataset, clip_model, loss_function, optimizer, epochs, device):
    for epoch in range(0, epochs):
        for images, queries in tqdm(training_dataset, desc=f"epoch {epoch}"):
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

