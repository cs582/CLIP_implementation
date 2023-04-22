# Senrich et. al 2016 & Gage et al. 1994
import os
import re
import math
import json
import tqdm
import pickle
import collections

from datetime import datetime as dt


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
    vocab = {'[EOS]': len(body_text), '[SOS]': len(body_text)}
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
    for sentence in tqdm.tqdm(body_text):
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


def check_saving_status(n_saved_vocabs, curr_size, frequency):
    return n_saved_vocabs == (curr_size//frequency-1)



class BytePairEncoderTokenizer:
    def __init__(self, vocab_size=1000, min_freq=2):
        self.max_vocab_size = vocab_size
        self.min_freq = min_freq
        self.token_ids = None
        self.vocab = None
        self.merges = None

    def train(self, body_text, filedir):
        # Initialize vocab with [SOS] and [EOS] tokens and single characters
        self.vocab = initialize_vocabulary(body_text)
        self.merges = {}

        # Initialize the corpus
        corpus = prepare_corpus(body_text)

        # Get the merges
        X = 3000
        threshold = X
        current_vocab_size = len(self.vocab)
        print(f"Retrieving vocabulary. curr_size {current_vocab_size}, max_size {self.max_vocab_size}")
        while current_vocab_size < self.max_vocab_size:
            pairs = get_status(corpus)
            bp = max(pairs, key=pairs.get)
            new_char = ''.join(bp)
            if pairs[bp] < self.min_freq:
                break

            # Update the vocabulary and BPE codes
            corpus = update_corpus(bp, corpus)
            self.vocab[new_char] = pairs[bp]
            self.merges[bp] = new_char

            if current_vocab_size > threshold:
                # Create TokenIDs
                print(f"Checkpoint: curr_size {current_vocab_size}, max_size {self.max_vocab_size}")
                token_map = {}
                for idx, word in enumerate(self.vocab.keys()):
                    token_map[word] = idx+1
                self.token_ids = token_map
                self.save_tokenizer(filedir)

                threshold += X

            current_vocab_size = len(self.vocab)

        # Create TokenIDs
        print("Creating TokenIDs...")
        token_map = {}
        for idx, word in enumerate(self.vocab.keys()):
            token_map[word] = idx+1
        self.token_ids = token_map
        self.save_tokenizer(filedir)


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

    def tokenize(self, sentence, remove_special_char=True, max_length=74):
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

        # Cut the sentence size of tokens length is larger than max_size
        # Else append zeros to reach max_size
        if len(tokens) < max_length:
            tokens = tokens + [0]*(max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        return tokens

    def save_tokenizer(self, filedir, test_mode=False):
        assert self.token_ids is not None, f"token_ids not found, load vocabulary from json file or train the model."
        assert self.vocab is not None, f"vocabulary not found, load vocabulary from json file or train the model."
        assert self.merges is not None, f"merges not found, load vocabulary from json file or train the model."

        print("Saving tokenizer...")
        if not os.path.exists(filedir):
            os.mkdir(filedir)

        filename = f"CLIP_text_tokenizer_{dt.strftime(dt.now(), '%Y%m%d_%H:%M:%S')}.pickle"

        if test_mode:
            filename = "TEST_" + filename

        filepath = f"{filedir}/{filename}"

        with open(filepath, "wb") as f:
            tokenizer_info = {
                "vocab": self.vocab,
                "merges": self.merges,
                "token_ids": self.token_ids
            }
            pickle.dump(tokenizer_info, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Successfully saved tokenizer as {filename}!!!")

    def load_tokenizer(self, filename):
        print("Loading tokenizer...")
        with open(filename, "rb") as f:
            tokenizer_info = pickle.load(f)
            self.vocab = tokenizer_info["vocab"]
            self.merges = tokenizer_info["merges"]
            self.token_ids = tokenizer_info["token_ids"]
            print(f"Successfully loaded tokenizer from {filename}!!!")