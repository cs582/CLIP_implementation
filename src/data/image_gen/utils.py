import os
import re
import json
import time
import requests
import numpy as np


# Global Variables Patterns
image_src_pattern = re.compile(r'srcSet="[^"]+"', re.DOTALL | re.IGNORECASE)
alt_pattern = re.compile(r'alt="[^"]+"', re.DOTALL | re.IGNORECASE)


def query(word, page):
    return f"https://unsplash.com/napi/search/photos?query={word}&per_page=30&page={page}&xp=search-quality-boosting%3Acontrol"


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


def make_GET_request(word, page):
    # Make GET Request
    start = time.time()
    response = requests.get(query(word, page))
    end = time.time()

    print(f"WORD: {word} PAGE: {page}. RESPONSE: {response.status_code}. TIME: {end-start} seconds")
    return response


def make_pair(full_info):
    new_dict = {}
    new_dict['query'] = full_info['alt_description']
    new_dict['image'] = full_info['urls']['small']
    return new_dict


def retrieve_pairs(words_file, from_ith_word=0, test_mode=False, load_from_checkpoint=False):
    path = "src/data/image_gen/pairs"

    # Filter out useless words
    words = sorted(filter_out_words(words_file))
    if test_mode:
        words = words[from_ith_word+3]

    # Total number of words
    num_words = len(words)

    # Initialize pairs
    pairs = []
    curr_n_pairs = 0

    # Iterate through each word
    for curr_word_number in range(from_ith_word, num_words):
        word = words[curr_word_number]

        # Make initial response to estimate number of pages
        response = make_GET_request(word, 1)

        # Proceed if STATUS CODE = 200
        if response.status_code == 200:
            # Store prev total number of pairs
            prev_n_pairs = curr_n_pairs

            # Load data into json file
            data_json = json.loads(response.text)

            # Obtain information about the query
            results = data_json['results']
            total_images_av = data_json['total']
            total_pages_av = data_json['total_pages']

            # Iterate through all pages
            for page in range(1, min(total_pages_av+1, 334)):
                if page > 1:
                    # Make GET Request
                    response = make_GET_request(word, page)

                    # Load data into json file
                    data_json = json.loads(response.text)

                    # Get results
                    results = data_json['results']

                # Get query image pairs for the current request
                new_pairs = [make_pair(res) for res in results]

                # Append new pairs to list of pairs
                pairs += new_pairs

            # Calculate number of pairs afterwards
            curr_n_pairs = len(pairs)

            # Calculate progress in %
            progress = 100 * np.round(curr_word_number / num_words, 4)

            # Image scraping loop message
            string = f"{curr_word_number}/{num_words} {progress}% | "
            string += f"Retrieved {curr_n_pairs-prev_n_pairs} of {total_images_av} for the word '{word}'"
            string += f" ::: N PAIRS is {curr_n_pairs}."
            print(string)

        # Save every 2% of the progress
        if num_words > 1000 and curr_word_number % (num_words//50) == 0:
            filename = f"{curr_word_number}th_word_part_{num_words//50}"
            save_pairs_to_json(pairs, path, filename)
            pairs = []

    return pairs