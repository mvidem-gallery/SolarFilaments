import yaml

from utils import get_project_root
from omegaconf import OmegaConf

config = yaml.load(open('../config.yaml'), Loader=yaml.FullLoader)
config = OmegaConf.create(config)

root = get_project_root() # a Path object
dataset_path = root / 'data' / 'MAGFiLO_1.0_Kaggle_2026'
path = {
    'data': {
        'train' : {
            'train_images_path' : dataset_path / 'train' / 'train_images',
            'labels_json_path' : dataset_path / 'train' / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json',
        },
        'test' : {
            'test_images_path' : dataset_path / 'test' / 'test_images_path',
        }
    },
    'weights': root / 'weights.pth'
}
path = OmegaConf.create(path)