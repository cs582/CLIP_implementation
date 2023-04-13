# Senrich et. al 2016 & Gage et al. 1994
import re, collections


def fix_sentence(sentence):
    if sentence[-1] == '.':
        return sentence[:-1]
    return sentence


def add_tags(sentence):
    return '[SOS]' + sentence + '[EOS]'


def initialize_vocabulary(body_text):
    vocab = {'[SOS]': len(body_text), '[EOS]': len(body_text)}
    for sentence in body_text:
        sentence = fix_sentence(sentence)
        for word in sentence.split():
            for char in word:
                vocab[char] = vocab.get(char, 0) + 1

    return vocab


def prepare_corpus(body_text):
    corpus = []
    for sentence in body_text:
        sentence = fix_sentence(sentence).split()
        for i, word in enumerate(sentence):
            if i == 0:
                w_out = ['[SOS]'] + [char for char in word] + ['</w>']
            elif i == len(sentence)-1:
                w_out = [char for char in word] + ['[EOS] </w>']
            else:
                w_out = [char for char in word] + ['</w>']
            corpus.append(' '.join(w_out))
    return corpus


def get_status(corpus):
    pairs = collections.defaultdict(int)
    for word in corpus:
        symbols = word.split()
        for i in range(len(symbols)-2):
            pairs[symbols[i], symbols[i+1]] = pairs.get((symbols[i], symbols[i+1]), 0) + 1
    return pairs


def merge_vocab(pair, corpus):
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
        self.bpe_codes = None

    def train(self, body_text):
        # Initialize vocab with [SOS], [EOS], and single characters
        self.vocab = initialize_vocabulary(body_text)

        corpus = prepare_corpus(body_text)

        while len(self.vocab.keys()) < self.vocab_size:
            pairs = get_status(corpus)
            bp = max(pairs, key=pairs.get)
            if pairs[bp] < self.min_freq:
                break
            corpus = merge_vocab(bp, corpus)
            self.vocab[''.join(bp)] = pairs[bp]

        print(self.vocab, len(self.vocab.keys()))


body_text = [
    "This is one sentence",
    "Suppose this is something smart.",
    "I don't think anybody would complain.",
    "What if we add",
    "A few sentences more."
]


# Create an instance of the BytePairEncoderTokenizer
tokenizer = BytePairEncoderTokenizer(vocab_size=100)
tokenizer.train(body_text)