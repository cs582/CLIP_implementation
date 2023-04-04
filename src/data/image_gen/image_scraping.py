import numpy as np
from utils import retrieve_pairs

filename = "src/data/nlp/words/words_snap_at_14840164_DONE.json"

pairs = retrieve_pairs(filename)

rd1, rd2, rd3 = np.random.randint(0, len(pairs)), np.random.randint(0, len(pairs)), np.random.randint(0, len(pairs))

print("img 0 src = ", pairs[rd1][0][:50], "...", pairs[rd1][0][-20:], "alt = ", pairs[rd1][1])
print("img 1 src = ", pairs[rd2][0][:50], "...", pairs[rd2][0][-20:], "alt = ", pairs[rd2][1])
print("img 2 src = ", pairs[rd2][0][:50], "...", pairs[rd3][0][-20:], "alt = ", pairs[rd3][1])