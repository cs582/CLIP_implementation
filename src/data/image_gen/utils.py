import os
import re
import json
import time
import requests
import numpy as np
from tqdm import tqdm
import urllib.request


# Global Variables Patterns
image_src_pattern = re.compile(r'srcSet="[^"]+"', re.DOTALL | re.IGNORECASE)
alt_pattern = re.compile(r'alt="[^"]+"', re.DOTALL | re.IGNORECASE)


def query(word):
    return f"https://unsplash.com/napi/search/photos?query={word}&per_page=MAX&page=1&xp=search-quality-boosting%3Acontrol"


def save_pairs_to_json(pairs, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)

    file = f"{path}/{filename}.json"
    with open(file, 'w') as f:
        json.dump(pairs, f)
        print(f"Successfully saved pairs as {file}")


def load_pairs_from_json(path, filename):
    file = f"{path}/{filename}.json"
    with open(file, 'r') as f:
        pairs = json.load(f)
    return pairs


def filter_out_words(words_file):
    # Matches only English words, words without numbers in between, and numbers with exactly 4 digits
    regex = r'\b[a-zA-Z]+\b|\b[A-Za-z]+\b'

    with open(words_file, 'r') as f:
        words = json.load(f)

    string_words = " ".join(words)
    filtered_words = re.findall(regex, string_words)

    initial_number = len(string_words.split(" "))
    final_number = len(filtered_words)

    print(f"Total words {initial_number} | After filtering out garbage words {final_number}.")

    return filtered_words


def retrieve_pairs(words_file, from_ith_word=0, test_mode=False):
    path = "src/data/image_gen/pairs"
    filename = f"{from_ith_word-1}_pairs"

    # Filter out useless words
    words = sorted(filter_out_words(words_file))
    if test_mode:
        words = words[from_ith_word:from_ith_word+3]
    else:
        words = words[from_ith_word:]

    # Total number of words
    num_words = len(words)

    # Initialize pairs
    pairs = []

    # Load pairs from json if starting from a word != 0
    if not test_mode and from_ith_word > 0:
        pairs = load_pairs_from_json(path, filename)

    # Image scraping loop
    curr_image_number = 0

    for word in words:
        # Make GET Request
        start = time.time()
        response = requests.get(query(word))
        end = time.time()

        print(f"WORD: {word}. RESPONSE: {response.status_code}. TIME: {end-start} seconds")

        if response.status_code == 200:
            # Store number of pairs before data retrieval
            prev_n_pairs = len(pairs)

            # Load data into json file
            data_json = json.loads(response.text)
            results = data_json['results']
            total_available = data_json['total']

            for result in results:
                new_dict = {}
                new_dict['query'] = result['alt_description']
                new_dict['image'] = result['urls']['small']
                pairs.append(new_dict)

            # Calculate number of pairs afterwards
            curr_n_pairs = len(pairs)

            # Calculate progress in %
            progress = 100 * np.round(curr_image_number / num_words, 4)

            # Image scraping loop message
            string = f"{curr_image_number}/{num_words} {progress}% | "
            string += f"Retrieved {curr_n_pairs-prev_n_pairs} of {total_available} for the word '{word}'"
            string += f" ::: N PAIRS is {curr_n_pairs}."
            print(string)

        # Save every 5% of the progress
        if num_words > 1000 and curr_image_number % (num_words//20) == 0:
            filename = f"{curr_image_number}_pairs"
            save_pairs_to_json(pairs, path, filename)

    return pairs