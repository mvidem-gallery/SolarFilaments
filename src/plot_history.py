import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe: saves to a file, doesn't need a display
import matplotlib.pyplot as plt

from config import path


def plot_history(history_path, output_path="training_curves.png"):
    with open(history_path) as f:
        history = json.load(f)

    epochs = history["epoch"]
    val_epochs = history.get("val_epoch", epochs)  # fallback for older history files without val_epoch
    has_val_iou = "val_iou" in history and len(history["val_iou"]) > 0

    n_plots = 2 if has_val_iou else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    axes = [axes] if n_plots == 1 else list(axes)

    # --- Loss curve ---
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    if "val_loss" in history and len(history["val_loss"]) > 0:
        axes[0].plot(val_epochs, history["val_loss"], label="Val Loss", marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- IoU curve (only if validation was used) ---
    if has_val_iou:
        axes[1].plot(val_epochs, history["val_iou"], label="Val IoU", color="green", marker="o")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("IoU")
        axes[1].set_title("Validation IoU Curve")
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves to '{output_path}'")


if __name__ == "__main__":
    default_history_path = os.path.splitext(str(path.weights))[0] + "_history.json"

    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default=default_history_path, help="Path to the training history JSON file")
    parser.add_argument("--output", default="training_curves.png", help="Where to save the resulting plot")
    args = parser.parse_args()

    plot_history(args.history, args.output)