import os
import cv2
import random
import zipfile

from utils import *

from kaggle.api.kaggle_api_extended import KaggleApi
from torch.utils.data import Dataset, DataLoader
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
        self.images_idx_coco = self.coco.getImgIds()


    def __getitem__(self, idx):
        idx_coco = self.images_idx_coco[idx]

        if self.mode == 'eval':
            image_path = os.path.join(self.current_images_path, str(self.images[idx]))
            image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            return image_gray, None, None

        try:
            metadata = self.coco.loadImgs(idx_coco)[0]
        except Exception:
            metadata = next((img for img in self.coco.dataset['images'] if img['id'] == idx_coco), None)

        image_path = os.path.join(self.current_images_path, metadata['file_name'])
        image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image_gray = self.transform(image_gray)

        if image_gray is None:
            raise FileNotFoundError(f"Couldn't open {image_path}. Mode: {self.mode}")

        try:
            ann_ids = self.coco.getAnnIds(imgIds=[idx_coco])
            anns = self.coco.loadAnns(ann_ids)
        except Exception:
            anns = [ann for ann in self.coco.dataset['annotations'] if ann['image_id'] == idx_coco]

        return image_gray, anns, metadata


    def __len__(self):
        return len(self.images)


    def get_random_image(self):
        idx = random.randint(0, len(self.images) - 1)
        return self[idx]


    def get_image_by_filename(self, filename : str):
        if filename not in self.images and filename not in self.images_path:
            raise FileNotFoundError(f'File {filename} not found in SolarDataset. Mode: {self.mode}')

        try:
            image_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        except Exception:
            image_path = os.path.join(self.current_images_path, filename)
            image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if self.mode == 'eval':
            return image_gray, None, None

        metadata = next((img for img in self.coco.dataset['images'] if img['file_name'] == filename), None)

        if self.transform:
            image_gray = self.transform(image_gray)

        try:
            ann_ids = self.coco.getAnnIds(imgIds=[metadata['id']])
            anns = self.coco.loadAnns(ann_ids)
        except Exception:
            anns = [ann for ann in self.coco.dataset['annotations'] if ann['image_id'] == metadata['id']]

        return image_gray, anns, metadata


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
