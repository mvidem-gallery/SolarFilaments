import os
import cv2
import json
import zipfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

        try:
            metadata = self.coco.loadImgs(idx_coco)[0]
        except Exception:
            metadata = next(img for img in self.coco.dataset['images'] if img['id'] == idx_coco)

        image_path = os.path.join(self.current_images_path, metadata['file_name'])
        image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image_gray = self.transform(image_gray)

        if image_gray is None:
            raise FileNotFoundError(f"Не вдалося відкрити {image_path}")

        try:
            ann_ids = self.coco.getAnnIds(imgIds=[idx_coco])
            anns = self.coco.loadAnns(ann_ids)
        except Exception:
            anns = [ann for ann in self.coco.dataset['annotations'] if ann['image_id'] == idx_coco]

        return image_gray, anns, metadata


    def __len__(self):
        return len(self.images)


    def train(self):
        self.mode = 'train'
        self.current_images_path = self.train_images_path


    def test(self):
        self.mode = 'test'
        self.current_images_path = self.test_images_path

# metadata = json.load(open(str(get_project_root() / 'data' \
#   / 'MAGFiLO_1.0_Kaggle_2026' / 'train' / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json')))
#
# print(metadata.keys())
# print(metadata['images'][1].keys())
# for i in range(100):
#     print(metadata['annotations'][i]['image_id'])

import random

# Шлях до JSON з анотаціями
ann_file = Path(get_project_root()) / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
coco = COCO(ann_file)

# Отримати всі image_ids (рядкові)
image_ids = coco.getImgIds()
img_id = random.choice(image_ids)

# Знайти інформацію про зображення вручну (бо id рядковий)
img_info = next(img for img in coco.dataset['images'] if img['id'] == img_id)

# Шлях до папки з картинками
img_dir = Path(get_project_root()) / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
img_path = img_dir / img_info['file_name']

# Завантажуємо картинку
image = plt.imread(img_path)

# Отримуємо анотації для цього зображення
ann_ids = coco.getAnnIds(imgIds=[img_id])
anns = coco.loadAnns(ann_ids)

# Відображаємо картинку
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(image)
ax.axis('off')

# Малюємо баундінг бокси
for ann in anns:
    if 'bbox' in ann:
        x, y, w, h = ann['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                 edgecolor='red', facecolor='none')
        ax.add_patch(rect)

plt.show()