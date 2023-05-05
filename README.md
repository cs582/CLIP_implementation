# Low Resource Implementation of Contrastive Language-Image Pre-Training (CLIP)
> From scratch implementation (at smaller scale due to limited resources) of CLIP. CLIP is an AI tool developed by OpenAI that connects images to text with zero-shot capabilities similar to those of GPT-2 and GPT-3. It uses Natural Language Processing for zero-shot classification.

![CUDA version][cuda-image]
![Python version][python-image]
![PyTorch version][pytorch-image]
![Einops version][einops-image]
![Scikit Learn version][scikit-learn-image]
![Matplotlib version][matplotlib-image]
![Pandas version][pandas-image]
![NumPy version][numpy-image]

This project implements the ground-breaking paper by OpenAI on
test-image connection and zero-shot classification: CLIP. This
paper is later important for image generation in DALL-E-2.

Based on the orignal work by OpenAI
> Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021, July). Learning transferable visual models from natural language supervision. In International conference on machine learning (pp. 8748-8763). PMLR.

## Installation

OS X & Linux:

```sh
pip install -r requirements
```

The largest model was trained on a single TITAN RTX.

The smallest models were trained on either
a single TITAN X or a single GTX 2080 Ti.

## Unit tests

Part of the development required testing different models, however,
as the modules grow larger and larger, and become more complex, I found
it necessary to implement unit tests to easily find if all modules,
starting from the smallest, were running correctly.

This tool also allowed me to benchmark different approaches and choose the
right one from several options.

Unit tests can be run by using the following command. ```cpu_heavy``` when set to
True, this will include CLIP and the full backbones (running on CPU) in the TestSuite; ```gpu_heavy```
when set to True, will include CLIP and the full backbones running on CUDA in the TestSuite. Lastly,
```test_n``` is set to 0 to run all tests, 1 to run only small modules and backbones
and then 2 to test only the full CLIP model.

```sh
python unit_tests.py -cpu_heavy=True -gpu_heavy=True -test_n=0
```

## Training

To train the machine learning model, you can choose from 2 Text-Encoders (Base and Large)
and 4 ViT models (Base/32 @ 226, Base/16 @ 112, Small/16 @ 112, Small/8 @ 112).

You can choose to do ```fine-tuning``` (limited to 100 steps), to choose your preferred
```device``` (either cpu or cuda), to load from the last checkpoint ```load_last_checkpoint``` which saves automatically
eavery epoch, choose the number of ```warmup``` steps.

Then you can also choose to tune the following hypper-parameters: ```temperature```, ```batch_size```,
```epochs```, ```vocab_size```, ```max_length``` (of the sentences), (weight) ```decay```, ```beta_1```,
```beta_2```, ```epsilon```, (learning rate) ```lr```, width of the text encoder ```text_dim_out```, 
width of the image encoder ```image_dim_out```, multi embedding dimension ```embedding_dim```

```sh
CLIP training cycle with evaluation.

optional arguments:
-h, --help            show this help message and exit
-fine_tuning FINE_TUNING
Perform Fine tuning over one epoch. Requires arg model
different from default:None.
-device DEVICE        Set device to use: gpu or cpu.
-load_last_checkpoint LOAD_LAST_CHECKPOINT
Load model from last checkpoint and restart training
from there.
-warmup WARMUP        Warmup steps.
-image_encoder IMAGE_ENCODER
Image encoder backbone. One of (ViT) @112, @224, or
@336.
-text_encoder TEXT_ENCODER
Text encoder backbone. One of S (Small), B (Base), or
L (Large).
-max_temperature MAX_TEMPERATURE
Maximum temperature for CLIP loss.
-batch_size BATCH_SIZE
Batch size. Is the same as the multimodal embedding
dimension.
-epochs EPOCHS        Epochs for training. (ignored in fine-tuning).
-vocab_size VOCAB_SIZE
Vocabulary size from trained tokenizer.
-max_length MAX_LENGTH
Max length of the token encoding.
-decay DECAY          Weight decay.
-beta_1 BETA_1        Adam optimizer beta_1.
-beta_2 BETA_2        Adam optimizer beta_2. Recommended 0.98 for ViT.
-epsilon EPSILON      Adam optimizer epsilon. Recommended 1e-6 for ViT.
-lr LR                Learning rate.
-text_dim_out TEXT_DIM_OUT
Text encoder output dimension.
-image_dim_out IMAGE_DIM_OUT
Image encoder output dimension.
-embedding_dim EMBEDDING_DIM
Embedding dimension CLIP.
```


```sh
python train.py
```


## About the Author

Carlos Gustavo Salas Flores – [LinkedIn](https://www.linkedin.com/in/carlosgustavosalas/) – yuseicarlos2560@gmail.com

Distributed under the MIT license. See ``LICENSE.txt`` for more information.

[https://github.com/cs582](https://github.com/cs582/)


<!-- Markdown link & img dfn's -->
[cuda-image]: https://img.shields.io/badge/CUDA-11.5-blue?style=flat-square]
[python-image]: https://img.shields.io/badge/Python-3.8.5-blue?style=flat-square]
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.9.0-orange?style=flat-square]
[einops-image]: https://img.shields.io/badge/Einops-1.0.0-orange?style=flat-square]
[scikit-learn-image]: https://img.shields.io/badge/scikit--learn-0.24.1-blue?style=flat-square]
[matplotlib-image]: https://img.shields.io/badge/Matplotlib-3.3.4-orange?style=flat-square]
[pandas-image]: https://img.shields.io/badge/Pandas-1.2.3-blue?style=flat-square]
[numpy-image]: https://img.shields.io/badge/NumPy-1.20.1-orange?style=flat-square]