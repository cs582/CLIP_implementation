import re
import os
import argparse

from src.data.data_utils import task1_join_json_files, task2_download_and_save_images, task3_5_queries_to_txt
from src.data import build_WKIT_dataset, build_imagenet_cifar_dataset, build_nft_dataset
from src.models.natural_language_processing.nlp_tokenization import train_bpe

parser = argparse.ArgumentParser(
    prog='CLIP Data.',
    description='CLIP data preparation.',
    epilog='The data preparation includes (1) reading the json files and converting them into a single csv file'\
    '(2) downloading all images from the csv file and labeling them to a local directory file'\
    '(3) training the tokenizer using the queries from task 1'
)

parser.add_argument('-task', type=float, default=2, help='Set data to perform task 1, 2, 3 (or 3.5), or 4. Read description for more info.')
parser.add_argument('-cap', type=int, default=10, help='Cap the number of images to download. Set to -1 for full length.')
parser.add_argument('-start', type=int, default=0, help='Starting image to save.')
parser.add_argument('-vocab_size', type=int, default=10000, help='Vocabulary size for task 3: training tokenizer.')

args = parser.parse_args()


if __name__ == "__main__":

    pairs_folder = "src/data/image_gen/WQ-dataset"
    tokenizer_folder = "src/data/nlp/tokenizers"
    images_folder = "/data/carlos"

    if not os.path.exists(tokenizer_folder):
        os.mkdir(tokenizer_folder)
    if not os.path.exists(pairs_folder):
        os.mkdir(pairs_folder)

    if args.task == 1:
        task1_join_json_files(pairs_folder)

    if args.task == 2:
        task2_download_and_save_images(pairs_folder, args)

    if args.task == 3.5:
        task3_5_queries_to_txt(pairs_folder, tokenizer_folder)

    if args.task == 3:
        train_bpe(tokenizer_folder)

    if args.task == 4:
        build_WKIT_dataset.build(images_folder)

    if args.task == 5:
        build_nft_dataset.build()

    if args.task == 6:
        build_imagenet_cifar_dataset.build("imagenet")

    if args.task == 7:
        build_imagenet_cifar_dataset.build("cifar10")

