import re
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from PIL import Image

pattern = re.compile("^[0-9]+\.jpg$")


def valid_image_checker(image_dir, x):
    try:
        return pattern.search(x) and min(Image.open(os.path.join(image_dir, x)).size) >= 112
    except:
        print(f"Error with image {x}")
        return False


def build(image_dir):
    wq_dataset_file = "src/data/image_gen/WQ-dataset/WKIT_dataset.csv"

    # Read csv file with queries
    print(f"Reading {wq_dataset_file}")
    queries = pd.read_csv(wq_dataset_file, index_col=0, usecols=['query']).index

    # Get all (valid) images from image directory
    img_in_dir = [x for x in tqdm(os.listdir(image_dir), total=len(image_dir), desc="Getting (valid) images") if valid_image_checker(image_dir, x)]

    for x in img_in_dir:
        try:
            int(x[:-4])
        except:
            print(f"{x} failed")

    # Valid queries mask
    idx = np.array([int(x[:-4]) for x in img_in_dir])

    # Queries
    queries = queries[idx].tolist()

    # Check both queries and images have the same size
    assert len(queries)==len(img_in_dir), f"Queries {len(queries)} size doesn't match images size {len(img_in_dir)}"

    # Put queries and images in the same list
    pairs = [[q, img] for q, img in zip(queries, img_in_dir)]

    # Save pairs in DataFrame
    wqi_local_file = "src/data/image_gen/WQ-dataset/WKIT_local.csv"
    print(f"Saving as {wqi_local_file}")
    pd.DataFrame(pairs, columns=["Q", "IMG"]).to_csv(wqi_local_file, index=False, header=True)
    print(f"Successfully saved WQI_local as {wqi_local_file}")