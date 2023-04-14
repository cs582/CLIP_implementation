import json
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def resize_image(image, x, y):
    height, width = image.shape[:2]
    ratio = max(x/width, y/height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image


class ImageQueryDataset(Dataset):
    def __init__(self, data_dir, filename, img_res=(224, 224)):
        self.img_res = img_res
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(img_res),
        ])

        # Get a list of image file paths
        with open(f"{data_dir}/{filename}", 'r') as f:
            self.image_files = json.load(f)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # Load the image at the given index
        image_path = self.image_files[index]
        image_full_path = f"{self.data_dir}/images/{image_path}"
        image = cv2.imread(image_full_path)

        if image.shape[0] < self.img_res[0] or image.shape[1] < self.img_res[1]:
            image = resize_image(image, *self.img_res)

        # Apply any data transformations if specified
        if self.transform is not None:
            image = self.transform(image)

        # Get the label from the image file name
        label = image_path.replace('.jpg', '').replace('_', ' ')

        return image, label
