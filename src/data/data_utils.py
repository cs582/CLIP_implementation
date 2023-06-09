import os
import io
import json
import pandas as pd
import numpy as np
import concurrent.futures

from tqdm import tqdm
from PIL import Image
from urllib.request import urlopen

# -----------------
# HELPERS
# -----------------


def url_image_save_multithreaded(urls, path, num_workers=20, first_index=0):
    """
    Download images in multithread from a list of urls.
    :param urls: (list) List of images' urls.
    :param path: (str) Directory to save the images.
    :param num_workers: (int) Maximum number of processes to run. Suggested number_of_cores + 4
    :param first_index: (int) First index.
    :return: List of local addresses of all downloaded images.
    """
    curr_index = first_index
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for url in tqdm(urls, desc="Stacking Processes"):
            name = f"{curr_index}"
            future = executor.submit(download_image_sync, url, path, name)
            futures.append(future)
            curr_index += 1
        results = [future.result() for future in tqdm(concurrent.futures.as_completed(futures), total=len(urls), desc="Downloading Images")]
        return results


def download_image_sync(url, path, name):
    """
    Helper function to download an image from a given URL using Pillow.
    :param url: (str) Image URL.
    :param path: (str) Local images address.
    :param name: (str) Image saving name.
    :return: (str) if successful else None
    """
    try:
        resp = urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        with Image.open(io.BytesIO(image)) as image:
            filepath = os.path.join(path, f"{name}.jpg")
            image.save(filepath)
            return f"{name}.jpg"
    except:
        print(f"Error while downloading image {url}")

# ----------------
# Task 1
# ----------------


def task1_join_json_files(pairs_folder):
    """
    Helper function to join all JSON files and store them in a csv named WKIT_24M_dataset.csv.
    :param pairs_folder: (str) folder with all images URLs and queries extracted in JSON files.
    :return: None
    """
    data_frames = []

    for file_name in os.listdir(pairs_folder):
        # If not a valid file then move on
        if "word" not in file_name:
            continue

        # Check current json file loading
        print("current file", file_name, "...")
        file = f"{pairs_folder}/{file_name}"
        with open(file, 'r') as f:
            data_temp = json.load(f)

        # Drop duplicates
        temp_df = pd.DataFrame(data_temp).dropna().drop_duplicates()

        # Add dataframe to list of dataframes
        data_frames.append(temp_df)

    # Concatenate dataframes
    data = pd.concat(data_frames, ignore_index=True).drop_duplicates()

    data.to_csv(f"{pairs_folder}/WKIT_24M_dataset.csv", header=True)

# ----------------
# Task 2
# ----------------


def task2_download_and_save_images(pairs_folder, args):
    """
    Download all images in WKIT_24M_dataset.csv and store them in a local directory.
    :param pairs_folder: (str) folder to store all pairs.
    :param args: (parser.args) console arguments.
    :return: None
    """
    # Get csv file address
    csv_filepath = f"{pairs_folder}/WKIT_24M_dataset.csv"

    # Read csv file
    print(f"reading {csv_filepath}")
    df = pd.read_csv(csv_filepath, index_col=0)
    df_row_len, df_col_len = df.shape

    # Get cap limit
    idx_0 = args.start
    idx_f = df_row_len if args.cap == -1 else args.cap

    # Extract queries and images addresses
    queries, img_address = df['query'].tolist()[idx_0:idx_f], df['image'].tolist()[idx_0:idx_f]

    # Download images and store them into a new directory
    folder = "images"
    images_dir = f"{pairs_folder}/{folder}"

    if not os.path.exists(images_dir):
        os.mkdir(images_dir)

    url_image_save_multithreaded(urls=img_address, path=images_dir, first_index=idx_0)


# ----------------
# Task 3
# ----------------


def task3_5_queries_to_txt(pairs_folder, tokenizer_folder):
    """
    Save all queries into a text file as a corpus for tokenizer.
    :param pairs_folder: (str) Local folder to store all pairs.
    :param tokenizer_folder: (str) Folder to store the corpus, same place as the tokenizer.
    :return:
    """
    # Get csv file address
    csv_filepath = f"{pairs_folder}/WKIT_24M_dataset.csv"

    # Read csv file
    print(f"reading {csv_filepath}")
    df = pd.read_csv(csv_filepath, index_col=0)

    # Get all queries
    body_text = "\n".join(df['query'].tolist())

    # File to save corpus
    filename_corpus = f'{tokenizer_folder}/corpus.txt'

    # Save corpus
    file = open(filename_corpus, "w")
    a = file.write(body_text)
    file.close()
    print(f"Saved corpus as {filename_corpus}")