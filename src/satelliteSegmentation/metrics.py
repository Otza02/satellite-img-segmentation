from satelliteSegmentation.dataset import SatelliteData

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.inference_mode()
def mean_iou_dice(
    model: nn.Module, loader: DataLoader, device: str, num_classes: int, eps: float = 1e-7
):
    model.eval()

    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        for cls in range(num_classes):
            pred_cls = preds == cls
            target_cls = masks == cls

            intersection[cls] += (pred_cls & target_cls).sum()
            union[cls] += (pred_cls | target_cls).sum()

    valid = union > 0
    iou = (intersection[valid] + eps) / (union[valid] + eps)
    dice = (2 * intersection[valid] + eps) / (union[valid] + eps)
    return iou.mean().item(), dice.mean().item()


@torch.inference_mode()
def pixel_accuracy(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    correct = 0
    total = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        correct += (preds == masks).sum().item()
        total += masks.numel()
    return correct / total


import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.inference_mode()
def segmentation_metrics(
    model: nn.Module, loader: DataLoader, device: str, num_classes: int, eps: float = 1e-7
):
    model.eval()

    conf_matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
        device=device,
    )

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        preds = preds.view(-1)
        masks = masks.view(-1)

        valid = (masks >= 0) & (masks < num_classes)

        preds = preds[valid]
        masks = masks[valid]

        indices = num_classes * masks + preds

        conf_matrix += torch.bincount(
            indices,
            minlength=num_classes**2,
        ).reshape(num_classes, num_classes)

    cm = conf_matrix.float()

    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    # Pixel Accuracy
    pixel_acc = tp.sum() / cm.sum()

    # IoU
    iou = tp / (tp + fp + fn + eps)

    # Dice
    dice = (2 * tp) / (2 * tp + fp + fn + eps)

    # Ignorar clases ausentes
    valid_classes = (tp + fp + fn) > 0

    miou = iou[valid_classes].mean()
    mean_dice = dice[valid_classes].mean()

    return {
        "dice": mean_dice.item(),
        "miou": miou.item(),
        "pixel_acc": pixel_acc.item(),
        "dice_per_class": dice.cpu().tolist(),
        "iou_per_class": iou.cpu().tolist(),
        "confusion_matrix": cm.cpu().numpy(),
    }

def main():
    from satelliteSegmentation.config import Config
    from satelliteSegmentation.models.unet import UNet
    from satelliteSegmentation.dataset import load_data
    
    conf = Config("cpu")
    model = UNet(conf)
    state = torch.load("checkpoints/baseline_1.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state)
    
    data = load_data("val")
    loader = DataLoader(data, 128)
    results = segmentation_metrics(model, loader, conf.device, conf.num_classes)
    print(results)

if __name__ == "__main__":
    main()