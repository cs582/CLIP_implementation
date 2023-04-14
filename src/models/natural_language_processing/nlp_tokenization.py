# Senrich et. al 2016 & Gage et al. 1994
import re, collections


def fix_sentence(sentence):
    if sentence[-1] == '.':
        return sentence[:-1]
    return sentence


def add_tags(sentence):
    return '[SOS]' + sentence + '[EOS]'


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
        self.vocab = None
        self.merges = None

    def train(self, body_text):
        # Initialize vocab with [SOS], [EOS], and single characters
        self.vocab = initialize_vocabulary(body_text)

        corpus = prepare_corpus(body_text)

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

    def encode(self, words):
        # Apply the trained BPE codes to encode a word
        if self.vocab is None:
            raise ValueError("Tokenizer has not been trained yet!")

        new_sentence = " ".join(words)
        for pair, new_char in self.merges.items():
            bigram = re.escape(' '.join(pair))
            p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
            new_sentence = p.sub(''.join(pair), new_sentence)

        encoded_sentence = new_sentence
        for tokenID, (symbol, _) in enumerate(self.vocab.items()):
            bigram = re.escape(symbol)
            p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
            encoded_sentence = p.sub(str(tokenID), encoded_sentence)

        return new_sentence, [int(x) for x in encoded_sentence.split()]

    def tokenize(self, sentence):
        # Tokenize a text using the trained BPE tokenizer
        if self.merges is None:
            raise ValueError("Tokenizer has not been trained yet!")

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