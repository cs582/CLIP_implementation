import tqdm
from torch.utils.data import DataLoader
from src.data.data_loader import ImageQueryDataset

def loading_WQI_images():
    dataset_file = "src/data/image_gen/WQ-dataset/WQI_local.csv"
    image_path = "/data/carlos/images"
    tokenizer_file = "src/data/nlp/tokenizers/CLIP-bpe.tokenizer.json"
    max_length = 32

    dataset = ImageQueryDataset(dataset_file, image_path, tokenizer_file, max_length, img_res=112)
    dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, num_workers=10, pin_memory=True, drop_last=True)

    for x, y in tqdm.tqdm(dataloader, desc="Sanity Check"):
        ""

