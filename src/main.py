import argparse
import os
import torch
import random

from utils import set_seed, data_exists
from config import config, path
from dataset import download_data, SolarDataset, build_dataloader, EvalDataset
from train import train, load_model, predict
from create_submission import create_submission
from visualize import visualize
from plot_history import plot_history

from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from augmentations import train_transform, val_transform, test_transform


def parse_args():
    parser = argparse.ArgumentParser(
        description="SolarFilaments pipeline. With no flags, runs training."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "-predict",
        action="store_true",
        help="Run sliding-window inference on the test set, saving predicted masks as PNGs.",
    )
    mode.add_argument(
        "-submission",
        action="store_true",
        help="Run inference on the test set and build submission.csv (per-filament RLE).",
    )
    mode.add_argument(
        "-visualize",
        action="store_true",
        help="Overlay saved predicted masks on the original test images for visual inspection.",
    )
    mode.add_argument(
        "-history",
        action="store_true",
        help="Plot the saved training loss/IoU curves to a PNG.",
    )

    # Only relevant together with -history.
    parser.add_argument(
        "--history-path",
        default=None,
        help="Path to the training history JSON (defaults to the file saved next to the model weights).",
    )
    parser.add_argument(
        "--output",
        default="training_curves.png",
        help="Output PNG path for -history (default: training_curves.png).",
    )

    return parser.parse_args()


def build_device():
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Project is heavy and requires cuda!")
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


if __name__ == "__main__":
    args = parse_args()
    set_seed(config.seed)

    # -visualize and -history only touch files on disk (saved masks /
    # saved training history) — no need to load the model or touch CUDA.
    if args.visualize:
        visualize()

    elif args.history:
        history_path = args.history_path or (
                os.path.splitext(str(path.weights))[0] + "_history.json"
        )
        plot_history(history_path, args.output)

    else:
        device = build_device()
        model = load_model(config, device)

        if args.predict:
            model.load_state_dict(torch.load(path.weights, map_location=device))

            test_dataset = EvalDataset(
                test_images_path=path.data.test.test_images_path,
                transform=test_transform,
            )

            predict(
                config=config,
                model=model,
                device=device,
                dataset=test_dataset,
                output_dir='predictions'
            )

        elif args.submission:
            model.load_state_dict(torch.load(path.weights, map_location=device))

            test_dataset = EvalDataset(
                test_images_path=path.data.test.test_images_path,
                transform=test_transform,
            )

            create_submission(
                model=model,
                device=device,
                dataset=test_dataset,
                num_classes=config.num_classes,
            )

        else:
            if not data_exists('data/'):
                download_data()

            coco = COCO(path.data.train.labels_json_path)
            all_img_ids = coco.getImgIds()
            random.shuffle(all_img_ids)

            split_idx = int(len(all_img_ids) * 0.8)
            train_ids = all_img_ids[:split_idx]
            val_ids = all_img_ids[split_idx:]

            # Sliding-window validation over the FULL val set is expensive
            # (see earlier discussion), so we validate on a fixed random
            # subset of unique images instead. Sampled once (with the
            # global seed) so the same images are used every validation
            # run within this training session, keeping metrics comparable
            # across epochs.
        if len(val_ids) > config.val_subset_size:
            val_ids = random.sample(val_ids, config.val_subset_size)

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

        # batch_size=1: validation images are now kept at full resolution
        # (no crop/resize) for sliding-window inference, so they can't
        # necessarily be stacked into a batch > 1.
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        train(
            config=config,
            model=model,
            path=path,
            device=device,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            val_every=config.val_every,
            val_last_n_epochs=config.val_last_n_epochs,
        )
