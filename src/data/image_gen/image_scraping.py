import numpy as np
from utils import retrieve_pairs

filename = "src/data/nlp/words/words_snap_at_14840164_DONE.json"

pairs = retrieve_pairs(filename)

rd1, rd2, rd3 = np.random.randint(0, len(pairs)), np.random.randint(0, len(pairs)), np.random.randint(0, len(pairs))

rd_pair1 = pairs[rd1]
rd_pair2 = pairs[rd2]
rd_pair3 = pairs[rd3]

print("img 0 link:", rd_pair1[0][:50], "...", rd_pair1[0][-20:], "query:", rd_pair1[1], "word:", rd_pair1[2])
print("img 1 link:", rd_pair2[0][:50], "...", rd_pair2[0][-20:], "query:", rd_pair2[1], "word:", rd_pair2[2])
print("img 2 link:", rd_pair3[0][:50], "...", rd_pair3[0][-20:], "query:", rd_pair3[1], "word:", rd_pair3[2])

print(f"DONE WITH {len(pairs)} PAIRS.")