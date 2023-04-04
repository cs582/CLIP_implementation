import re
import json
import numpy as np
from tqdm import tqdm
import urllib.request


# Global Variables Patterns
image_src_pattern = re.compile(r'srcSet="[^"]+"', re.DOTALL | re.IGNORECASE)
alt_pattern = re.compile(r'alt="[^"]+"', re.DOTALL | re.IGNORECASE)


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


def get_img_tags_from_with_label(webpage, label):
    try:
        url = f"{webpage}/{label}"
        response = urllib.request.urlopen(url) # Download the HTML
        html = response.read().decode() # Decode the HTML from bytes to string
        #print("response received")
    except:
        #print("Not able to retrieve data from", url)
        return []

    # Focus solely on the grid requested
    pattern = re.compile(r'<div\s+data-test="search-photos-route">.*<div\s+class="dvFt2">', re.DOTALL | re.IGNORECASE)
    string = pattern.findall(html)[0]

    #print("string searching over", string[:25], "...", string[-25:])

    # Match all img tags
    pattern = re.compile(r'<img[^>]+>', re.DOTALL | re.IGNORECASE)
    img_tags = pattern.findall(string)

    # Remove all content inside HTML tags from the body HTML using the regex pattern
    return img_tags


def get_img_label_pairs(img_tag):
    # Get Image
    try:
        # Stay only with the first img hyperlink
        img_src = image_src_pattern.findall(img_tag)[0]
        # Split the string in case there are many images and just keep the first one
        img_src = img_src.split(" ")[0]
        # Delete the first part ' srcSet=" ' of the string
        img_src = img_src[8:]
    except:
        return "", ""

    try:
        alt_text = alt_pattern.findall(img_tag)[0]
        # Delete the first part ' alt=" '  and the last ' " ' of the string
        alt_text = alt_text[5:-1]
    except:
        return "", ""

    return img_src, alt_text



def retrieve_pairs(words_file):
    # Filter out useless words
    words = sorted(filter_out_words(words_file))[:2]

    # Total number of words
    num_words = len(words)

    # Set page to extract images from
    page = "https://unsplash.com/s/photos"

    pairs = []

    # Image scraping loop
    curr_image_number = 0
    for word in words:
        prev_pairs_length = len(pairs)
        curr_image_number += 1

        # Retrieve img tags
        img_tags = get_img_tags_from_with_label(webpage=page, label=word)

        # Retrieve the image sources and alt text from the img tag
        for img_tag in img_tags:
            img, label = get_img_label_pairs(img_tag)
            if len(img) > 1:
                pairs.append((img, label))

        progress = np.round(100*curr_image_number/len(words), 3)

        curr_pairs_length = len(pairs)

        # Image scraping loop message
        string = f"{curr_image_number}/{num_words} {progress}"
        string += f"{len(img_tags)} img tags, but only {prev_pairs_length-curr_pairs_length} "
        string += f"valid image/query pairs found for word {word}"
        string += f" ::: Total pairs so far {curr_pairs_length}"
        print(string)



    return pairs