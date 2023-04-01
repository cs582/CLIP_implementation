import urllib.request
import collections

url = "https://en.wikipedia.org/wiki/OpenAI"
response = urllib.request.urlopen(url) # Download the HTML
html = response.read().decode() # Decode the HTML from bytes to string

# Tokenize the HTML into words and count the number of occurrences of each word
word_counts = collections.Counter(html.split())

# Print the word counts
for word, count in word_counts.items():
    print(word, ":", count)
