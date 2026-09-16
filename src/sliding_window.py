import math

import numpy as np
import torch
import torch.nn.functional as F


def _padded_size(size, tile_size, stride):
    if size <= tile_size:
        return tile_size
    n_steps = math.ceil((size - tile_size) / stride)
    return tile_size + n_steps * stride


def _tile_positions(padded_size, tile_size, stride):
    positions = list(range(0, padded_size - tile_size + 1, stride))
    if positions[-1] != padded_size - tile_size:
        positions.append(padded_size - tile_size)
    return positions


def sliding_window_logits(model, image, device, tile_size=512, stride=384, num_classes=2, batch_size=4):
    """
    Runs `model` over a full-resolution image using overlapping tiles and
    stitches the tile logits back into one full-size logits map, averaging
    predictions in the overlap regions.

    Why this matters: resizing a large image down to 512x512 (the old
    approach) shrinks thin filaments to a handful of pixels or erases them
    entirely. Tiling lets the model see every region at its native scale.

    image: torch.FloatTensor [C, H, W], already normalized the same way as
           training (e.g. via A.Normalize()). Can be on CPU or `device`.
    Returns: torch.FloatTensor [num_classes, H, W] on `device`, cropped
             back to the original (unpadded) H, W.
    """
    model.eval()
    C, H, W = image.shape

    padded_h = _padded_size(H, tile_size, stride)
    padded_w = _padded_size(W, tile_size, stride)
    padded = F.pad(image, (0, padded_w - W, 0, padded_h - H), mode="constant", value=0)

    y_positions = _tile_positions(padded_h, tile_size, stride)
    x_positions = _tile_positions(padded_w, tile_size, stride)
    tile_coords = [(y, x) for y in y_positions for x in x_positions]

    accum_logits = torch.zeros((num_classes, padded_h, padded_w), device=device)
    counts = torch.zeros((1, padded_h, padded_w), device=device)

    with torch.no_grad():
        for start in range(0, len(tile_coords), batch_size):
            batch_coords = tile_coords[start:start + batch_size]
            tiles = torch.stack([
                padded[:, y:y + tile_size, x:x + tile_size] for y, x in batch_coords
            ]).to(device)

            logits = model(tiles)["out"]  # [B, num_classes, tile_size, tile_size]

            for logit, (y, x) in zip(logits, batch_coords):
                accum_logits[:, y:y + tile_size, x:x + tile_size] += logit
                counts[:, y:y + tile_size, x:x + tile_size] += 1

    accum_logits /= counts.clamp(min=1)
    return accum_logits[:, :H, :W]


def sliding_window_predict(model, image, device, tile_size=512, stride=384, num_classes=2, batch_size=4):
    """Same as sliding_window_logits, but returns the final class-index map."""
    logits = sliding_window_logits(model, image, device, tile_size, stride, num_classes, batch_size)
    preds = torch.argmax(logits, dim=0)
    return preds.cpu().numpy().astype(np.uint8)
