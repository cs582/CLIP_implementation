import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def display_images(images, image_names=None):

    ncols = 3
    nrows = len(images) // ncols + 1

    print(ncols, nrows)

    if type(images[0]) == torch.Tensor:
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15, 15))

        curr_index = 0
        for i in range(nrows):
            for j in range(ncols):

                if curr_index==len(images):
                    break

                ax[i, j].imshow(images[curr_index].permute(1, 2, 0))
                if image_names is None:
                    ax[i, j].set_title(f"Image {curr_index}")
                else:
                    ax[i, j].set_title(image_names[curr_index])
                curr_index += 1

    elif type(images[0]) == str:

        fig, ax = plt.subplots(ncols=ncols, nrows=nrows)

        curr_index = 0
        for i in range(ncols):
            for j in range(nrows):
                x = Image.open(images[curr_index])
                ax[i, j].imshow(x)
                if image_names is None:
                    ax[i, j].set_title(f"Image {curr_index}")
                else:
                    ax[i, j].set_title(image_names[curr_index])
                curr_index += 1

    plt.show()


def display_logits_heatmap(logits, images, text_labels):
    # Code originally writen by OpenAI (github.com/openai/CLIP)
    count = len(text_labels)

    plt.figure(figsize=(20,14))

    plt.title("Text-Image cosine similarities", size=20)
    plt.imshow(logits, vmin=0.0, vmax=0.3)
    plt.yticks(range(len(text_labels)), text_labels, fontsize=18)
    plt.xticks([])

    for i, image in enumerate(images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin='lower')

    for x in range(logits.shape[1]):
        for y in range(logits.shape[0]):
            plt.text(x, y, f"{logits[y, x]:.2f}", ha="center", va="center", size=12)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])

    return