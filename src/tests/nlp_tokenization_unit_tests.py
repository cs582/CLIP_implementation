import unittest
import time

from eval_loop.models.natural_language_processing.nlp_tokenization import BytePairEncoderTokenizer


class TokenizationUnitTest(unittest.TestCase):
    def test_tokenization(self):
        tokenizer = BytePairEncoderTokenizer(vocab_size=100, min_freq=2)

        body_text = [
            "This is the test for the Byte Pair Encoder Tokenizer",
            "CLIP is an AI developed by OpenAI",
            "It is used for text and image feature extraction in Dalle-2",
            "But is also used in other algorithms f.e. Stable Diffusion 1 and 2",
            "It was also implemented fr0m scratch, hopefully, it works just fine."
            "We hope this is a good test case."
        ]

        start = time.time()
        tokenizer.train(body_text)
        end = time.time()

        test_sentence = f"Dogs, Cats, and Programmers. Were not part of the corpus."

        tokens = tokenizer.tokenize(test_sentence)

        # Check that all tokens are integers
        for token in tokens:
            self.assertEqual(type(token), int)

        print("tokenized: ", test_sentence, " -> ", tokens)
        print(f"Tokenizer training done in {end-start} seconds.")

    def test_saving_tokenizer(self):

        tokenizer = BytePairEncoderTokenizer(vocab_size=100, min_freq=2)

        body_text = [
            "This is the test for the Byte Pair Encoder Tokenizer",
            "CLIP is an AI developed by OpenAI",
            "It is used for text and image feature extraction in Dalle-2",
            "But is also used in other algorithms f.e. Stable Diffusion 1 and 2",
            "It was also implemented fr0m scratch, hopefully, it works just fine."
            "We hope this is a good test case."
        ]

        tokenizer.train(body_text)
        tokenizer.save_tokenizer("src/data/nlp/tokenizers", test_mode=True)



