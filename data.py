import re
import os
import cv2
import json
import argparse
import pandas as pd

from urllib.request import urlopen
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    prog='CLIP Data.',
    description='CLIP data preparation.',
    epilog='The data preparation includes (1) reading the json files and converting them into a single csv file'\
    '(2) downloading all images from the csv file and labeling them to a local directory file'
)

parser.add_argument('-task', type=int, default=2, help='Set data to perform task 1 or 2. Read description for more info.')
parser.add_argument('-cap', type=int, default=10, help='Cap the number of images to download.')

args = parser.parse_args()


def url_image_save(url, path, name):
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, -1)

    filepath = f"{path}/{name}.jpg"
    cv2.imwrite(filepath, image)
    cv2.waitKey(0)
    return filepath


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

        cap = args.cap

        # Extract queries and images addresses
        queries, img_address = df['query'].tolist()[:cap], df['image'].tolist()[:cap]

        # Download images and store them into a new directory
        folder = "images"
        images_dir = f"{pairs_folder}/{folder}"

        # Create new folder if doesn't exist
        if not os.path.exists(images_dir):
            os.mkdir(images_dir)

        # Save images addresses into a list
        image_queries = []
        for url, q in zip(img_address, queries):
            q = clean_sentence(q).replace(" ", "_")
            url_image_save(url, images_dir, q)

        # Save list to json file
        images_json_file = f"{pairs_folder}/image-queries-cap-at-{cap}.json"
        with open(images_json_file, "w") as f:
            json.dump(image_queries, f)
            print(f"images saves successfully as {images_json_file}")

