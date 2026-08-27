import torch
import random
from pycocotools.coco import COCO
from utils import set_seed, data_exists
from config import config, path
from dataset import download_data, SolarDataset, build_dataloader
from train import train, load_model
from augmentations import train_transform, val_transform

if __name__ == "__main__":
    set_seed(config.seed)

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Project is heavy and requires cuda!")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = load_model(config, device)

    if not data_exists('data/'):
        download_data()

    coco = COCO(path.data.train.labels_json_path)
    all_img_ids = coco.getImgIds()
    random.shuffle(all_img_ids)

    split_idx = int(len(all_img_ids) * 0.8)
    train_ids = all_img_ids[:split_idx]
    val_ids = all_img_ids[split_idx:]

    train_dataset = SolarDataset(
        transform=train_transform,
        valid_ids=train_ids,
        **path.data.train
    )
    train_dataloader = build_dataloader(train_dataset, config)

    val_dataset = SolarDataset(
        transform=val_transform,
        train_images_path=path.data.train.train_images_path,
        labels_json_path=path.data.train.labels_json_path,
        valid_ids=val_ids,
        mode='val'
    )

    # val_config = config.dataloader.copy()
    # val_config['shuffle'] = False
    val_dataloader = build_dataloader(val_dataset, config)

    train(
        config=config,
        model=model,
        path=path,
        device=device,
        train_loader=train_dataloader,
        val_loader=val_dataloader
    )
