import os
from glob import glob

import cv2
import numpy as np

from config import path

TEST_IMAGES_DIR = path.data.test.test_images_path
PREDICTIONS_DIR = "predictions"
OUTPUT_DIR = "visualizations"

MASK_COLOR = (0, 0, 255)  # red, in BGR
ALPHA = 0.45  # mask opacity


def overlay_mask(image_gray, mask, color=MASK_COLOR, alpha=ALPHA):
    """Blends a colored mask on top of a grayscale image wherever mask > 0."""
    image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

    colored_mask = np.zeros_like(image_bgr)
    colored_mask[mask > 0] = color

    blended = cv2.addWeighted(image_bgr, 1 - alpha, colored_mask, alpha, 0)
    overlay = image_bgr.copy()
    overlay[mask > 0] = blended[mask > 0]

    return overlay


def visualize(test_images_dir=TEST_IMAGES_DIR, predictions_dir=PREDICTIONS_DIR, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(glob(os.path.join(str(test_images_dir), "*.jpeg")))
    if not image_paths:
        raise RuntimeError(f"No test images found in {test_images_dir}")

    saved = 0
    for image_path in image_paths:
        file_name = os.path.basename(image_path)
        mask_path = os.path.join(predictions_dir, os.path.splitext(file_name)[0] + "_mask.png")

        if not os.path.exists(mask_path):
            print(f"No prediction found for {file_name}, skipping")
            continue

        image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Predictions were made on a resized (512x512) image, so scale the
        # mask back up to the original resolution before overlaying.
        if mask.shape != image_gray.shape:
            mask = cv2.resize(
                mask,
                (image_gray.shape[1], image_gray.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        overlay = overlay_mask(image_gray, mask)

        side_by_side = np.hstack([
            cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR),
            overlay,
        ])

        out_name = os.path.splitext(file_name)[0] + "_overlay.png"
        cv2.imwrite(os.path.join(output_dir, out_name), side_by_side)
        saved += 1

    print(f"Saved {saved} visualizations to '{output_dir}'")


if __name__ == "__main__":
    visualize()
