import urllib.request
import re
from lxml import etree
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
        print(f"Retrieved {len(words)}. Remaining {len(filtered_words)} filtered words. Page: {url}")

        return filtered_words

    except:
        print("Not able to retrieve data from", url)
        return []


def read_dump_file(dump_file):
    # create an XML parser
    parser = etree.iterparse(dump_file, events=("start", "end"))

    # iterate through the XML elements
    for event, elem in parser:
        # check if the element is a page
        if event == "end" and elem.tag.endswith("page"):
            # extract the title of the page
            title = elem.findtext("{http://www.mediawiki.org/xml/export-0.10/}title")
            # do something with the title
            print(title)

            # clear the element to save memory
            elem.clear()
            break