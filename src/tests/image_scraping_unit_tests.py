import unittest
import numpy as np

from eval_loop.data.image_gen.utils import retrieve_pairs


class ImageScrappingUnitTests(unittest.TestCase):
    def test_retrieve_query_image_pairs(self):
        filename = "src/data/nlp/words/words_snap_at_14840164_DONE.json"
        starting_word = 100

        pairs = retrieve_pairs(filename, from_ith_word=starting_word, test_mode=True)

        for img_link, q, w in pairs:
            if np.random.rand() < 0.05:
                print("img 0 link:", img_link[:50], "...", img_link[-20:], "query:", q, "word:", w)

        print(f"DONE WITH {len(pairs)} PAIRS.")