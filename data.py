import re
import os
import cv2
import json
import asyncio
import argparse
import pandas as pd

import concurrent.futures
import aiohttp
from urllib.request import urlopen, Request
from tqdm import tqdm
from PIL import Image
import io

from src.models.natural_language_processing.nlp_tokenization import BytePairEncoderTokenizer

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    prog='CLIP Data.',
    description='CLIP data preparation.',
    epilog='The data preparation includes (1) reading the json files and converting them into a single csv file'\
    '(2) downloading all images from the csv file and labeling them to a local directory file'\
    '(3) training the tokenizer using the queries from task 1'
)

parser.add_argument('-task', type=int, default=2, help='Set data to perform task 1, 2, or 3. Read description for more info.')
parser.add_argument('-cap', type=int, default=10, help='Cap the number of images to download. Set to -1 for full length.')
parser.add_argument('-start', type=int, default=0, help='Starting image to save.')
parser.add_argument('-vocab_size', type=int, default=10000, help='Vocabulary size for task 3: training tokenizer.')

args = parser.parse_args()

# def url_image_save(url, path, name):
#     resp = urlopen(url)
#     image = np.asarray(bytearray(resp.read()), dtype="uint8")
#     image = cv2.imdecode(image, -1)
#
#     filepath = f"{path}/{name}.jpg"
#     cv2.imwrite(filepath, image)
#     cv2.waitKey(0)
#     return f"{name}.jpg"


async def download_image(session, url, path, name):
    try:
        async with session.get(url) as response:
            content = await response.read()
            with Image.open(io.BytesIO(content)) as image:
                filepath = os.path.join(path, f"{name}.jpg")
                image.save(filepath)
                return f"{name}.jpg"
    except:
        print(f"Error while downloading image {url}.")


async def url_image_save_async(urls, path, num_workers=10, first_index=0):
    img_index = first_index
    os.makedirs(path, exist_ok=True)
    async with aiohttp.ClientSession() as session:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            loop = asyncio.get_event_loop()
            tasks = []
            for url in urls:
                name = f"{img_index}"
                task = loop.run_in_executor(executor, download_image, session, url, path, name)
                tasks.append(task)
                img_index += 1
            results = await asyncio.gather(*tasks)
            return results


def clean_sentence(sentence):
    # Remove leading and trailing white spaces
    sentence = sentence.strip()

    # Remove extra spaces between words
    sentence = re.sub(r'\s+', ' ', sentence)

    # Handle special cases like "word1,word2" or "word1.word2"
    sentence = re.sub(r'(?<=[^\s])[.,;:?!]+(?=[^\s])', ' ', sentence)

    # Handle cases like "word1.word2" or "word1-word2"
    sentence = re.sub(r'(?<=\S)[.-](?=\S)', ' ', sentence)

    # Remove extra spaces after special characters
    sentence = re.sub(r'\s+([.,;:?!])', r'\1', sentence)

    return sentence



if __name__ == "__main__":

    pairs_folder = "src/data/image_gen/WQ-dataset"
    tokenizer_folder = "src/data/nlp/tokenizers"

    if args.task == 1:
        data = None
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

        data.to_csv(f"{pairs_folder}/WQI_mini.csv", header=True)

    if args.task == 2:
        # Get csv file address
        csv_filepath = f"{pairs_folder}/WQI_test.csv"

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

        # Save images
        # for idx, (url, q) in enumerate(zip(img_address, queries)):
        #     if idx < args.start:
        #         continue
        #     try:
        #         #img_dir = url_image_save(url, images_dir, idx)
        #     except:
        #         print(f"url: {url} failed.")

        loop = asyncio.get_event_loop()
        loop.run_until_complete(url_image_save_async(urls=img_address, path=images_dir, first_index=idx_0))


    if args.task == 3:
        # Get csv file address
        csv_filepath = f"{pairs_folder}/WQI_mini.csv"

        # Read csv file
        print(f"reading {csv_filepath}")
        df = pd.read_csv(csv_filepath, index_col=0)

        # Get all queries
        body_text = df['query'].tolist()

        # Initialize tokenizer
        tokenizer = BytePairEncoderTokenizer(vocab_size=args.vocab_size, min_freq=2)

        # Training tokenizer
        print(f"training tokenizer over {len(body_text)} queries")
        tokenizer.train(body_text)

        # Saving tokenizer vocabulary, merges, and token_ids
        tokenizer.save_tokenizer(tokenizer_folder)


