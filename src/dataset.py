import os
import cv2
import random
import zipfile
import torch
import numpy as np

from utils import *

from kaggle.api.kaggle_api_extended import KaggleApi
from torch.utils.data import Dataset
from glob import glob
from pycocotools.coco import COCO


def download_data():
    data_dir_path = 'data'
    if data_exists(data_dir_path):
        raise RuntimeError('Data already exists')

    root_path = get_project_root()
    download_path = root_path / data_dir_path

    os.makedirs(download_path, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    competition = "filament-segmentation-2026"

    api.competition_download_files(competition, path=str(download_path))

    with zipfile.ZipFile(str(download_path / 'filament-segmentation-2026.zip'), 'r') as zip_ref:
        zip_ref.extractall(str(download_path))

    os.remove(str(download_path / 'filament-segmentation-2026.zip'))



class SolarDataset(Dataset):
    """
    Dataset class for solar dataset.

    Contains each image from both train and test dirs.
    Anso contains COCO-style JSON file (to efficiently process
    COCO-format, pycocotools lib was used; structure of the file can be
    observed in https://www.kaggle.com/competitions/filament-segmentation-2026/data).
    """
    def __init__(
            self,
            train_images_path : str | Path,
            labels_json_path : str | Path,
            test_images_path : str | Path,
            mode : str = 'train',
            transform=None):
        self.train_images_path = train_images_path
        self.labels_json_path = labels_json_path
        self.test_images_path = test_images_path
        self.transform = transform
        self.mode = mode

        self.current_images_path = train_images_path if self.mode == 'train' else test_images_path
        self.images_path = glob(os.path.join(self.current_images_path, '*.jpeg'))
        self.images = sorted(os.path.basename(p) for p in self.images_path)

        self.coco = COCO(labels_json_path)
        print("Images in JSON:", len(self.coco.dataset.get('images', [])))
        print("Annotations in JSON:", len(self.coco.dataset.get('annotations', [])))
        print("Categories in JSON:", len(self.coco.dataset.get('categories', [])))
        self.images_idx_coco = self.coco.getImgIds()


    def __getitem__(self, idx):
        if self.mode == 'eval':
            filename = self.images[idx]

            image = cv2.imread(os.path.join(self.current_images_path, str(filename)), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Could not load test image: {filename}")

            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                image = torch.from_numpy(image).float().unsqueeze(0) / 255.0

            return image, None, {'file_name': filename}

        idx_coco = self.images_idx_coco[idx]
        try:
            metadata = self.coco.loadImgs(idx_coco)[0]
        except Exception:
            metadata = next((img for img in self.coco.dataset['images'] if img['id'] == idx_coco), None)

        image_path = os.path.join(self.current_images_path, metadata['file_name'])
        image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        mask = self._generate_mask(metadata, idx_coco)

        if self.transform:
            augmented = self.transform(image=image_gray, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = torch.from_numpy(image_gray).float().unsqueeze(0) / 255.0
            mask = torch.from_numpy(mask).long()

        if isinstance(mask, torch.Tensor):
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).float()
            elif mask.ndim == 3 and mask.shape[0] != 1:
                mask = mask.permute(2, 0, 1).float()

        return image, mask, metadata


    def __len__(self):
        return len(self.images)


    def _generate_mask(self, meta, idx):
        mask = np.zeros([meta['height'], meta['width']], dtype=np.uint8)
        anns_ids = self.coco.getAnnIds(imgIds=[idx])
        anns = self.coco.loadAnns(anns_ids)
        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann))

        return mask


    def get_random_image(self):
        idx = random.randint(0, len(self.images) - 1)
        return self[idx]


    def get_image_by_filename(self, filename: str):
        metadata = next((img for img in self.coco.dataset['images'] if img['file_name'] == filename), None)
        if metadata is None:
            raise FileNotFoundError(f"Image {filename} not found in COCO annotations.")

        idx = self.images_idx_coco.index(metadata['id'])
        return self[idx]


    def train(self):
        self.mode = 'train'
        self.current_images_path = self.train_images_path
        self.images_path = glob(os.path.join(self.current_images_path, '*.jpeg'))
        self.images = sorted(os.path.basename(p) for p in self.images_path)


    def eval(self):
        self.mode = 'eval'
        self.current_images_path = self.test_images_path
        self.images_path = glob(os.path.join(self.current_images_path, '*.jpeg'))
        self.images = sorted(os.path.basename(p) for p in self.images_path)
#
#
# root = get_project_root()
# pipe = root / 'data' / 'MAGFiLO_1.0_Kaggle_2026'
# train = pipe / 'train' / 'train_images'
# elabels = pipe / 'train' / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json'
# test = pipe / 'test' / 'test_images'
#
# dataset = SolarDataset(
#     train_images_path=train,
#     test_images_path=test,
#     labels_json_path=elabels,
# )
#
# image, mask, meta = dataset.get_image_by_filename('20110109104734Ch.jpeg')
#
# img = (image.squeeze(0).numpy() * 255).astype(np.uint8)
# msk = (mask.squeeze(0).numpy() * 255).astype(np.uint8)
#
# # Перетворюємо grayscale у BGR
# img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#
# # Накладаємо маску червоним
# img_bgr[msk > 0] = [0, 0, 255]
#
# cv2.imshow("Image with mask", img_bgr)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# print("meta file:", meta.get('file_name'))
# print("image shape:", img.shape)
# print("mask shape:", msk.shape)
# print("mask unique values:", np.unique(msk))
# print("mask sum:", msk.sum())

