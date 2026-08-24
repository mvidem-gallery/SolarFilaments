import yaml

from utils import get_project_root
from omegaconf import OmegaConf

config = yaml.load(open('../config.yaml'), Loader=yaml.FullLoader)
config = OmegaConf.create(config)

root = get_project_root() # Path object
dataset_path = root / 'data' / 'MAGFiLO_1.0_Kaggle_2026'
path = dict(
    data=dict(
        train_dir=dataset_path / 'train',
        test_dir=dataset_path / 'test',
        train_images=dataset_path / 'train' / 'train_images',
        test_images=dataset_path / 'test' / 'test_images',
        labels=dataset_path / 'train' / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json',
    ),
    weights=root / 'weights.pth'
)
path = OmegaConf.create(path)