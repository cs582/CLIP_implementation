from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


def train_bpe(tokenizer_folder):
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.BPEDecoder()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    # Define special tokens
    special_tokens = ['[EOS]', '[SOS]']

    # Add special tokens to the vocabulary
    tokenizer.add_tokens(special_tokens)

    # And then train
    trainer = trainers.BpeTrainer(
        vocab_size=43000,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train([f'{tokenizer_folder}/corpus.txt'], trainer=trainer)

    # And Save it
    tokenizer.save(f"{tokenizer_folder}/CLIP-bpe.tokenizer.json", pretty=True)