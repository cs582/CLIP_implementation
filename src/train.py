def training(training_dataset, clip_model, loss_function, optimizer, epochs):
    for epoch in range(0, epochs):
        print(f"Current epoch {epoch}...")
        for images, queries in training_dataset:
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

