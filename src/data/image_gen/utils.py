import re
import json
from tqdm import tqdm
import urllib.request


def get_img_tags_from_with_label(webpage, label):
    try:
        url = f"{webpage}/{label}"

        response = urllib.request.urlopen(url) # Download the HTML
        html = response.read().decode() # Decode the HTML from bytes to string

        # Define the regex pattern to match all words in the body
        pattern = re.compile(r'<body.*?>(.*?)</body>', re.DOTALL | re.IGNORECASE)
        body_html = pattern.search(html).group(1)

        # Match everything inside an img tag
        pattern = re.compile(r'<img(?=[^>]*>)((?:[^>=]|=(?:"[^"]*"|\'[^\']*\'|[^>\'"\s]*))*)>', re.IGNORECASE)
        img_tags = pattern.findall(body_html)

        # Remove all content inside HTML tags from the body HTML using the regex pattern
        return img_tags

    except:
        print("Not able to retrieve data from", url)
        return []


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


def get_pairs_from_img_tags(img_tags):
    pairs = []
    for tag in img_tags:
        # Match photo
        img_ref_pattern = re.compile(r'".*plus\.unsplash\.com.*?"')
        img_links = img_ref_pattern.findall(tag)

        if len(img_links) == 0:
            break

        # Match all alt attributes in the string
        alt_pattern = re.compile(r'title="([^"]*)"')
        alt_attributes = re.findall(alt_pattern, tag)

        pairs.append((alt_attributes, img_links))

    return pairs

def retrieve_image_label_pairs(words_file):

    words = sorted(filter_out_words(words_file))[:2]

    page = "https://unsplash.com/s/photos"

    pairs = []

    for word in tqdm(words, total=len(words)):
        img_tags = get_img_tags_from_with_label(webpage=page, label=word)
        curr_word_pairs = get_pairs_from_img_tags(img_tags)

        print(f"{len(img_tags)} img tags found found")

        if len(curr_word_pairs) > 0:
            print(curr_word_pairs[0])
            pairs.append(curr_word_pairs)
            print(f"{len(curr_word_pairs)} pairs found")

    return pairs