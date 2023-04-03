import torch
from einops.layers.torch import Rearrange

def my_images_to_pathces_implementation(img, batch_size, c, h, w, p):
    n_rows = h // p
    n_cols = w // p

    n = n_rows*n_cols

    x = torch.zeros(batch_size, n, c*p*p)

    idx = 0
    for i in range(n_cols):
        for j in range(n_rows):
            low_x, high_x = i*p, (i+1)*p
            low_y, high_y = j*p, (j+1)*p
            x[:, idx, :] = img[:, :, low_y: high_y, low_x:high_x].flatten(start_dim=1)
            idx += 1

    return x


def optimized_images_to_patches_implementation(imgs, batch_size, c, h, w, p):
    # Use the unfold function to extract patches from the images tensor
    patches = torch.nn.functional.unfold(imgs, kernel_size=p, stride=p)

    # Reshape the patches tensor to a 3D tensor with shape (batch_size, n_patches, patch_size)
    n_patches = patches.shape[2]
    patches = patches.transpose(1, 2).reshape(batch_size, n_patches, -1)

    return patches


def eignops_images_to_patches_optimization(imgs, batch_size, c, h, w, p):
    return Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)(imgs)
