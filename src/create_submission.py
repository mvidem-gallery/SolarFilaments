import csv
import os

import cv2
import numpy as np
from pycocotools import mask as maskUtils
from tqdm import tqdm

from sliding_window import sliding_window_predict

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


def create_submission(model, device, dataset, output_csv="submission.csv", tile_size=512, stride=384, num_classes=2):
    """
    dataset: an EvalDataset instance, iterated one image at a time (not
    batched) since test images can have different resolutions and
    sliding-window inference naturally works image-by-image.
    """
    model.eval()
    rows = []

    for i in tqdm(range(len(dataset)), desc="Predicting"):
        image, meta = dataset[i]
        image = image.to(device)

        # Sliding-window inference already returns the mask at the
        # image's ORIGINAL resolution, so no resize-back step is needed.
        mask = sliding_window_predict(
            model, image, device,
            tile_size=tile_size, stride=stride, num_classes=num_classes,
        )

        stem = os.path.splitext(meta["file_name"])[0]
        for idx, instance_mask in enumerate(split_instances(mask), start=1):
            filament_id = f"{stem}_{idx}"
            rows.append((filament_id, mask_to_rle(instance_mask)))

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} filament rows to '{output_csv}'")