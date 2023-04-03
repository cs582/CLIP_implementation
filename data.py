from src.create_dataset_utils import retrieve_image_label_pairs

filename = "data/nlp/words/words_snap_at_14840164_DONE.json"

pairs = retrieve_image_label_pairs(filename)

print(pairs[:10])

