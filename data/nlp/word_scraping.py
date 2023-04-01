import json
import nltk
import pandas as pd
from utils import get_words_from

def wikipedia_page(name):
    return f"https://en.wikipedia.org/wiki/{name}".replace(" ", "_")

nltk.download('stopwords')

# CSV with the most viewed articles
file = "data/nlp/top-wikipedia-articles-2023_02.csv"

# Get all most searched wikipedia pages urls
pages_names = pd.read_csv(file)['Page'].tolist()
pages_urls = [wikipedia_page(x) for x in pages_names]

# Initialize word counts dictionary
word_counts = {}

# Count all occurrences of words in all pages
for page_url in pages_urls[:1]:
    words = get_words_from(page_url)
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

# Keep only those words that appeared at least 100 times
useful_words = []
for key in word_counts.keys():
    if word_counts[key] >= 25:
        useful_words.append(key)

# Save useful words into a json file
with open("data/nlp/words.json", "w") as f:
    json.dump(useful_words, f)

# Final console message
print(f"finished scrappring with {len(useful_words)} words.")

