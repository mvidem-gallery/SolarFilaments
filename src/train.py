import json
import os
import cv2

import torch
import torch.nn as nn

from tqdm import tqdm
from torchvision import models
from torch.optim import Adam, lr_scheduler

from sliding_window import sliding_window_logits, sliding_window_predict


def load_model(config, device):
    model = models.segmentation.deeplabv3_resnet50(pretrained=True)

    old_conv = model.backbone.conv1
    model.backbone.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )

    with torch.no_grad():
        model.backbone.conv1.weight[:, 0, :, :] = torch.sum(old_conv.weight, dim=1)

    model.classifier[4] = nn.Conv2d(256, config.num_classes, kernel_size=1)

    return model.to(device)


class DiceLoss(nn.Module):
    """
    Soft Dice loss over all classes (softmax probabilities vs one-hot targets).
    Complements CrossEntropyLoss, which alone tends to be dominated by the
    background class when the foreground (filaments) covers a small fraction
    of pixels.
    """

    def __init__(self, num_classes, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_onehot = nn.functional.one_hot(targets, num_classes=self.num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


class CEDiceLoss(nn.Module):
    """CrossEntropy + Dice combined loss."""

    def __init__(self, num_classes, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)


def compute_iou(preds, targets, num_classes, eps=1e-7):
    """
    Mean IoU over foreground classes (class 0 is treated as background and
    excluded, since it would otherwise dominate and hide poor filament
    segmentation behind a trivially high background IoU).
    """
    preds = preds.view(-1)
    targets = targets.view(-1)

    ious = []
    for cls in range(1, num_classes):
        pred_mask = preds == cls
        target_mask = targets == cls

        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()

        if union == 0:
            continue  # class absent from both pred and target in this batch

        ious.append(intersection / (union + eps))

    return sum(ious) / len(ious) if ious else 0.0


def train(config, model, path, device, train_loader, val_loader=None, val_every=2, val_last_n_epochs=5):
    """
    val_every: validate every `val_every` epochs (e.g. 2 = every 2nd epoch).
    val_last_n_epochs: always validate during the final N epochs, regardless
        of `val_every`, so the final checkpoint decision is based on the
        freshest possible metric.
    """

    criterion = CEDiceLoss(num_classes=config.num_classes)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    # Sliding-window tile size / stride for full-image validation. Override
    # via config.tile_size / config.tile_stride if present.
    tile_size = config.get("tile_size", 512)
    tile_stride = config.get("tile_stride", 384)

    num_epochs = config.num_epochs
    best_val_iou = -1.0
    best_train_loss = float("inf")
    best_model_path = path.weights

    # Path where per-epoch loss/IoU history is saved, for later plotting
    # with plot_history.py. Kept next to the model weights.
    history_path = os.path.splitext(str(path.weights))[0] + "_history.json"
    history = {"epoch": [], "train_loss": []}
    if val_loader is not None:
        # val_epoch tracks WHICH epochs actually ran validation, since that's
        # no longer every epoch — needed to align val_loss/val_iou on plots.
        history["val_epoch"] = []
        history["val_loss"] = []
        history["val_iou"] = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        with tqdm(total=len(train_loader.dataset), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for images, masks, _ in train_loader:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)["out"]
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * images.size(0)
                pbar.update(images.size(0))

        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}")

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(epoch_loss)

        epoch_num = epoch + 1
        in_final_stretch = epoch_num > num_epochs - val_last_n_epochs
        should_validate = val_loader is not None and (epoch_num % val_every == 0 or in_final_stretch)

        if should_validate:
            val_loss, val_iou = evaluate(model, val_loader, criterion, device, config.num_classes, tile_size, tile_stride)
            scheduler.step(val_loss)
            print(f"Epoch {epoch_num}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

            history["val_epoch"].append(epoch_num)
            history["val_loss"].append(val_loss)
            history["val_iou"].append(val_iou)

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model updated at epoch {epoch_num}, val_iou={val_iou:.4f}")
        elif val_loader is not None:
            # Not a scheduled validation epoch: step the scheduler on train
            # loss instead so the LR can still react between the sparser
            # validation checkpoints.
            scheduler.step(epoch_loss)
            print(f"Epoch {epoch_num}: validation skipped (runs every {val_every} epochs, always in the last {val_last_n_epochs})")
        else:
            scheduler.step(epoch_loss)
            if epoch_loss < best_train_loss:
                best_train_loss = epoch_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model updated at epoch {epoch_num}, train_loss={epoch_loss:.4f}")

        # Written every epoch (not just at the end) so history survives an
        # interrupted run.
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"Training history saved to '{history_path}'")


def evaluate(model, val_loader, criterion, device, num_classes, tile_size=512, stride=384):
    """
    NOTE: val_loader must yield batch_size=1 (one full-resolution image per
    batch) — validation images are no longer cropped/resized, so they can
    have different sizes and can't be stacked into a larger batch.
    """
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    num_images = 0

    with torch.no_grad():
        for images, masks, _ in tqdm(val_loader, desc="Validating", leave=False):
            image = images[0].to(device)          # [C, H, W]
            mask = masks[0].to(device).long()      # [H, W]

            logits = sliding_window_logits(model, image, device, tile_size, stride, num_classes)

            loss = criterion(logits.unsqueeze(0), mask.unsqueeze(0))
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=0)
            total_iou += compute_iou(preds, mask, num_classes)
            num_images += 1

    return total_loss / max(num_images, 1), total_iou / max(num_images, 1)


def predict(config, model, device, dataset, output_dir, tile_size=512, stride=384):
    """
    Runs sliding-window inference over each full-resolution test image and
    saves the predicted mask as a PNG next to the original file name.

    dataset: an EvalDataset instance, iterated one image at a time (not
    batched) since test images can have different resolutions.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # Scale class indices (0..num_classes-1) up to the full 0-255 range so
    # the saved masks are actually visible when opened as images.
    scale = 255 // max(config.num_classes - 1, 1)

    saved = 0
    for i in tqdm(range(len(dataset)), desc="Predicting"):
        image, meta = dataset[i]
        image = image.to(device)

        pred_mask = sliding_window_predict(
            model, image, device,
            tile_size=tile_size, stride=stride, num_classes=config.num_classes,
        )

        out_name = os.path.splitext(meta["file_name"])[0] + "_mask.png"
        cv2.imwrite(os.path.join(output_dir, out_name), pred_mask * scale)
        saved += 1

    print(f"Saved {saved} predicted masks to '{output_dir}'")