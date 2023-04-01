import urllib.request
import re
import nltk
from nltk.corpus import stopwords

def remove_stopwords(word_list):
    # Get the set of English stop words
    stop_words = set(stopwords.words('english'))

    # Remove stop words from the list of words
    filtered_words = [word for word in word_list if not word in stop_words]

    return filtered_words


def get_words_from(url):
    try:
        response = urllib.request.urlopen(url) # Download the HTML
        html = response.read().decode() # Decode the HTML from bytes to string

        # Define the regex pattern to match all words in the body
        pattern = re.compile(r'<body.*?>(.*?)</body>', re.DOTALL | re.IGNORECASE)
        body_html = pattern.search(html).group(1)

        # Extract all words from the body HTML using the regex pattern
        words = re.findall(r'\b\w+\b', body_html)

        # Filter words
        filtered_words = remove_stopwords(words)

        # Print the words
        print(f"Retrieved {len(words)}. Remaining {len(filtered_words)} filtered words. Page: {url}")

        return filtered_words

    except:
        print("Not able to retrieve data from", url)
        return []


