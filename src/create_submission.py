import csv
import os

import cv2
import numpy as np
import torch
from pycocotools import mask as maskUtils
from tqdm import tqdm

from augmentations import val_transform
from config import config, path
from dataset import EvalDataset, build_eval_dataloader
from train import load_model

MIN_INSTANCE_AREA = 5  # drop tiny noise blobs when splitting into instances


def mask_to_rle(binary_mask):
    """Encodes a single binary instance mask into a COCO-style compressed RLE string."""
    rle = maskUtils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle["counts"]


def split_instances(binary_mask, min_area=MIN_INSTANCE_AREA):
    """
    The model predicts one semantic (filament / background) mask per image,
    but the submission needs one row per individual filament. We split the
    mask into separate instances using connected components.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    instances = []
    for label_id in range(1, num_labels):  # label 0 is background
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        instances.append((labels == label_id).astype(np.uint8))

    return instances


def create_submission(model, device, loader, original_sizes, output_csv="submission.csv"):
    model.eval()
    rows = []

    with torch.no_grad():
        for images, metas in tqdm(loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)["out"]
            preds = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)

            file_names = metas["file_name"]
            for pred_mask, file_name in zip(preds, file_names):
                orig_h, orig_w = original_sizes[file_name]
                # Model runs on a resized (512x512) input, so scale the
                # prediction back up to the original resolution first.
                mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                stem = os.path.splitext(file_name)[0]
                for i, instance_mask in enumerate(split_instances(mask), start=1):
                    filament_id = f"{stem}_{i}"
                    rows.append((filament_id, mask_to_rle(instance_mask)))

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} filament rows to '{output_csv}'")


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = load_model(config, device)
    model.load_state_dict(torch.load(path.weights, map_location=device))

    test_dataset = EvalDataset(
        test_images_path=path.data.test.test_images_path,
        transform=val_transform,
    )
    test_loader = build_eval_dataloader(test_dataset, config)

    # Original resolutions are needed to resize predictions back before
    # encoding RLE, since inference itself runs on resized 512x512 images.
    original_sizes = {}
    for file_name in test_dataset.images:
        img_path = os.path.join(str(test_dataset.test_images_path), file_name)
        h, w = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).shape
        original_sizes[file_name] = (h, w)

    create_submission(model, device, test_loader, original_sizes)
