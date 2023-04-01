import json
import nltk
import pandas as pd
from utils import extract_words_from_pages_in_dump_file

# Download stopwords
nltk.download('stopwords')

# CSV with the most viewed articles
file = "data/nlp/wkpages/enwiki-latest-pages-articles-multistream10.xml-p4045403p5399366"

# Extract words from pages in the dumpfile
words = extract_words_from_pages_in_dump_file(file)

# Final console message
print(f"finished scrapping with {len(words)} words.")

