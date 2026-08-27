import torch
import torch.nn as nn

import torchvision

from tqdm import tqdm
from torchvision import models, transforms
from torch.optim import Adam, lr_scheduler

import torch
import torch.nn as nn
from torchvision import models


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


def train(config, model, path, device, train_loader, val_loader=None):

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    num_epochs = config.num_epochs
    best_val_loss = float("inf")   # початкове значення
    best_model_path = path.weights

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
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model updated at epoch {epoch+1}, val_loss={val_loss:.4f}")
        else:
            scheduler.step(epoch_loss)
            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model updated at epoch {epoch+1}, train_loss={epoch_loss:.4f}")


def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, masks, _ in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)

    val_loss /= len(val_loader.dataset)
    return val_loss