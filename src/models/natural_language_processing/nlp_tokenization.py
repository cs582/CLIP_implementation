# Senrich et. al 2016 & Gage et al. 1994
import re, collections


def fix_sentence(sentence):
    if sentence[-1] == '.':
        return sentence[:-1]
    return sentence


def add_tags(sentence):
    return '[SOS]' + sentence + '[EOS]'


def remove_special_characters(text):
    """
    Removes all special characters from a string.
    """
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern, '', text)
    return text


def initialize_vocabulary(body_text):
    vocab = {'[EOS]':len(body_text), '[SOS]': len(body_text)}
    for sentence in body_text:
        sentence = fix_sentence(sentence)
        for word in sentence.split():
            for char in word:
                vocab[char] = vocab.get(char, 0) + 1

    return vocab


def sentence_to_symbol_format(sentence):
    words_out = []
    sentence = fix_sentence(sentence).split()
    for i, word in enumerate(sentence):
        w_out = [char for char in word] + ['</w>']
        words_out.append(' '.join(w_out))
    return words_out

def prepare_corpus(body_text):
    corpus = []
    for sentence in body_text:
        corpus += sentence_to_symbol_format(sentence)
    return corpus


def get_status(corpus):
    pairs = collections.defaultdict(int)
    for word in corpus:
        symbols = word.split()
        for i in range(len(symbols)-2):
            pairs[symbols[i], symbols[i+1]] = pairs.get((symbols[i], symbols[i+1]), 0) + 1
    return pairs


def update_corpus(pair, corpus):
    new_corpus = [None] * len(corpus)
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for i, word in enumerate(corpus):
        w_out = p.sub(''.join(pair), word)
        new_corpus[i] = w_out
    return new_corpus


class BytePairEncoderTokenizer:
    def __init__(self, vocab_size=1000, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.token_ids = None
        self.vocab = None
        self.merges = None

    def train(self, body_text):
        # Initialize vocab with [SOS], [EOS], and single characters
        self.vocab = initialize_vocabulary(body_text)

        # Initialize the corpus
        corpus = prepare_corpus(body_text)

        # Get the merges
        self.merges = {}
        while len(self.vocab.keys()) < self.vocab_size:
            pairs = get_status(corpus)
            bp = max(pairs, key=pairs.get)
            new_char = ''.join(bp)
            if pairs[bp] < self.min_freq:
                break
            # Update the vocabulary and BPE codes
            corpus = update_corpus(bp, corpus)
            self.vocab[new_char] = pairs[bp]
            self.merges[bp] = new_char

        # Create TokenIDs
        token_map = {}
        for idx, word in enumerate(self.vocab.keys()):
            token_map[word] = idx

        self.token_ids = token_map

    def encode(self, words):
        # Apply the trained BPE codes to encode a word
        if self.vocab is None:
            raise ValueError("Tokenizer has not been trained yet!")

        # Merge characters in the new sentence
        new_sentence = " ".join(words)
        for pair, new_char in self.merges.items():
            bigram = re.escape(' '.join(pair))
            p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
            new_sentence = p.sub(''.join(pair), new_sentence)

        # Encode tokens
        encoded_sentence = []
        for symbol in new_sentence.split():
            encoded_sentence.append(self.token_ids[symbol])
        return new_sentence, encoded_sentence

    def tokenize(self, sentence, remove_special_char=True):
        # Tokenize a text using the trained BPE tokenizer
        if self.merges is None:
            raise ValueError("Tokenizer has not been trained yet!")

        # Remove all special characters
        if remove_special_char:
            sentence = remove_special_characters(sentence)

        # Initialize the sentence into symbols of one character each
        sentence_words = []
        sentence = fix_sentence(sentence).split()
        for i, word in enumerate(sentence):
            char_out = [char for char in word]
            if i == 0:
                char_out = ['[SOS]'] + char_out
            elif i == len(sentence)-1:
                char_out = char_out + ['[EOS]']
            w_out = ' '.join(char_out)
            sentence_words.append(w_out)
        _, tokens = self.encode(sentence_words)
        return tokens