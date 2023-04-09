import numpy as np
from utils import retrieve_pairs

filename = "src/data/nlp/words/words_snap_at_14840164_DONE.json"

pairs = retrieve_pairs(filename, from_ith_word=0)

print(f"DONE WITH {len(pairs)} PAIRS.")