import urllib.request
import os
import re
import json
import tqdm
import numpy as np
from lxml import etree
from nltk.corpus import stopwords


def wikipedia_page(name):
    return f"https://en.wikipedia.org/wiki/{name}".replace(" ", "_")


def save_to_json(word_counts, path, file_name):
    # Keep only those words that appeared at least 100 times
    useful_words = []
    for key in word_counts.keys():
        if word_counts[key] >= 100:
            useful_words.append(key)

    if not os.path.exists(path):
        os.makedirs(path)

    file1 = f"{path}/words_{file_name}"
    file2 = f"{path}/counts_{file_name}"

    with open(file1, "w") as f:
        json.dump(useful_words, f)
        print(f"Successfully saved json file as {file1}")

    with open(file2, "w") as f:
        json.dump(word_counts, f)
        print(f"Successfully saved json file as {file2}")


def remove_stopwords(word_list):
    # Get the set of English stop words
    stop_words = set(stopwords.words('english'))

    # Remove stop words from the list of words
    filtered_words = [word for word in word_list if not word in stop_words]

    return filtered_words


def get_words_from(url, index_number, total):
    try:
        response = urllib.request.urlopen(url) # Download the HTML
        html = response.read().decode() # Decode the HTML from bytes to string

        # Define the regex pattern to match all words in the body
        pattern = re.compile(r'<body.*?>(.*?)</body>', re.DOTALL | re.IGNORECASE)
        body_html = pattern.search(html).group(1)

        # Remove all content inside HTML tags from the body HTML using the regex pattern
        body_text = re.sub(r'<[^>]*>|\b\w*(?:_\w+|\w*[A-Z]\w*)\b', '', body_html)

        # Remove all HTML syntax words from the body text using the regex pattern
        syntax_words = ['id', 'class', 'style', 'href', 'src', 'alt', 'title', 'rel', 'type', 'li']
        syntax_regex = r'\b(' + '|'.join(syntax_words) + r')\b'
        body_text = re.sub(syntax_regex, '', body_text)

        # Extract all words from the body text using the regex pattern
        words = re.findall(r'\b\w+\b', body_text)

        # Filter words
        filtered_words = remove_stopwords(words)

        # Print the words
        print(f"{np.round(index_number/total, 4)} : Retrieved {len(words)}. Remaining {len(filtered_words)} filtered words. Page: {url}")

        return filtered_words

    except:
        print("Not able to retrieve data from", url)
        return []


def extract_words_from_pages_in_dump_file(dump_file):
    # Initialize word counter
    word_counts = {}

    # create an XML parser
    parser = etree.iterparse(dump_file, events=("start", "end"))

    # Length of parser file
    size = 14840164

    # iterate through the XML elements
    index_number = 0
    for event, elem in parser:
        # check if the element is a page
        if event == "end" and elem.tag.endswith("page"):
            # extract the title of the page
            title = elem.findtext("{http://www.mediawiki.org/xml/export-0.10/}title")

            # Get wikipedia title
            page_url = wikipedia_page(title)

            # Get words from wikipedia page
            words = get_words_from(page_url, index_number, size)

            # Count word occurrence
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            # clear the element to save memory
            elem.clear()

        # Update index counter
        index_number += 1

        if index_number % 10000 == 0:
            save_to_json(word_counts, "data/nlp/words", f"snap_at_{index_number}.json")

