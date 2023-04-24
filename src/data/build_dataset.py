import os
import pandas as pd
import numpy as np

from PIL import Image


def build():
    wq_dataset_file = "src/data/image_gen/WQ-dataset/WQI_mini.csv"
    image_dir = "src/data/image_gen/WQ-dataset/images"

    # Read csv file with queries
    queries = pd.read_csv(wq_dataset_file, index_col=0, usecols=['query']).index

    # Get all (valid) images from image directory
    img_in_dir = [x for x in os.listdir(image_dir) if ".png" in x and min(Image.open(os.path.join(image_dir, x)).size) >= 112]

    # Valid queries mask
    idx = np.array([int(x[:-4]) for x in img_in_dir])

    # Queries
    queries = queries[idx].tolist()

    # Check both queries and images have the same size
    assert len(queries)==len(img_in_dir), f"Queries {len(queries)} size doesn't match images size {len(img_in_dir)}"

    # Put queries and images in the same list
    pairs = [[q, img] for q, img in zip(queries, img_in_dir)]

    # Save pairs in DataFrame
    wqi_local_file = "src/data/image_gen/WQ-dataset/WQI_local.csv"
    pd.DataFrame(pairs, columns=["Q", "IMG"]).to_csv(wqi_local_file)
    print(f"Successfully saved WQI_local as {wqi_local_file}")
