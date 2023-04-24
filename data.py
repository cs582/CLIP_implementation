import re
import os
import io
import json
import time
import argparse
import pandas as pd
import numpy as np
import concurrent.futures

from tqdm import tqdm
from PIL import Image
from urllib.request import urlopen
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


parser = argparse.ArgumentParser(
    prog='CLIP Data.',
    description='CLIP data preparation.',
    epilog='The data preparation includes (1) reading the json files and converting them into a single csv file'\
    '(2) downloading all images from the csv file and labeling them to a local directory file'\
    '(3) training the tokenizer using the queries from task 1'
)

parser.add_argument('-task', type=float, default=2, help='Set data to perform task 1, 2, or 3 (or 3.5). Read description for more info.')
parser.add_argument('-cap', type=int, default=10, help='Cap the number of images to download. Set to -1 for full length.')
parser.add_argument('-start', type=int, default=0, help='Starting image to save.')
parser.add_argument('-vocab_size', type=int, default=10000, help='Vocabulary size for task 3: training tokenizer.')

args = parser.parse_args()


def url_image_save_multithreaded(urls, path, num_workers=10, first_index=0):
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
    try:
        resp = urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        with Image.open(io.BytesIO(image)) as image:
            filepath = os.path.join(path, f"{name}.jpg")
            image.save(filepath)
            return f"{name}.jpg"
    except:
        print(f"Error while downloading image {url}")


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

    if not os.path.exists(tokenizer_folder):
        os.mkdir(tokenizer_folder)
    if not os.path.exists(pairs_folder):
        os.mkdir(pairs_folder)

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
        csv_filepath = f"{pairs_folder}/WQI_mini.csv"

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

    if args.task == 3.5:
        # Get csv file address
        csv_filepath = f"{pairs_folder}/WQI_mini.csv"

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

    if args.task == 3:
        # Initialize a tokenizer
        tokenizer = Tokenizer(models.BPE())

        # Customize pre-tokenization and decoding
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.BPEDecoder()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

        # Define special tokens
        special_tokens = ['[EOS]', '[SOS]']

        # Add special tokens to the vocabulary
        tokenizer.add_tokens(special_tokens)

        # And then train
        trainer = trainers.BpeTrainer(
            vocab_size=43000,
            min_frequency=2,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train([f'{tokenizer_folder}/corpus.txt'], trainer=trainer)

        # And Save it
        tokenizer.save(f"{tokenizer_folder}/CLIP-bpe.tokenizer.json", pretty=True)


