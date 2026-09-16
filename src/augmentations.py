import albumentations as A

from config import config
from albumentations.pytorch import ToTensorV2

CROP_SIZE = config.crop_size

# Training augmentations: sample a random high-resolution PATCH instead of
# squashing the whole image down to 512x512 with Resize(). Filaments are
# thin structures — resizing a large source image down shrinks them to only
# a few pixels wide (or erases them), which was likely the main reason the
# submission score was weak. Cropping preserves native pixel scale.
train_transform = A.Compose([
    A.PadIfNeeded(min_height=CROP_SIZE, min_width=CROP_SIZE, border_mode=0),
    A.RandomCrop(CROP_SIZE, CROP_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.CLAHE(p=0.8),
    A.Normalize(),
    ToTensorV2()
])

# Full-resolution transform used for VALIDATION during training. No crop or
# resize here — sliding_window_inference handles tiling itself, so the full
# image (and full mask) is used to compute a real, whole-image IoU/loss.
val_transform = A.Compose([
    A.Normalize(),
    ToTensorV2()
])

# Full-resolution transform used at TEST/submission time (predict.py /
# create_submission.py). Same reasoning as val_transform: no resize, since
# sliding-window inference covers the whole image via overlapping tiles.
test_transform = A.Compose([
    A.Normalize(),
    ToTensorV2()
])