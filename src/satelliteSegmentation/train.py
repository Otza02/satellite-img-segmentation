import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.amp.grad_scaler import GradScaler

from satelliteSegmentation.config import Config

from copy import deepcopy
from time import time


def run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer | None,
    scaler: GradScaler,
    config: Config,
):
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    for images, masks in dataloader:
        images = images.to(config.device, non_blocking=True)
        masks = masks.to(config.device, non_blocking=True)

        if is_training:
            optimizer.zero_grad()

        with torch.enable_grad() if is_training else torch.inference_mode():
            with torch.autocast(device_type=config.device, dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, masks)

            if not torch.isfinite(loss):
                # Un batch cuyos targets son enteramente ignore_index deja a
                # CrossEntropyLoss(reduction="mean") sin elementos válidos y
                # devuelve NaN. Se descarta el batch (sin backward/optimizer.step)
                continue

            if is_training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # loss.backward()
                # optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples if total_samples > 0 else float("nan")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    config: Config,
):
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scaler = GradScaler(config.device)

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")
    patience_count = 0
    best_model = None
    for epoch in range(config.epochs):
        start = time()
        train_loss = run_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
        )

        val_loss = run_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            optimizer=None,
            scaler=scaler,
            config=config,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time() - start
        print(
            f"Epoch {epoch + 1:02d}/{config.epochs} | "
            f"train_loss={train_loss:<6.4f} | "
            f"val_loss={val_loss:<6.4f} | "
            f"time {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        )

        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            patience_count = 0
            best_model = deepcopy(model.state_dict())
        else:
            patience_count += 1
            if patience_count >= config.patience:
                print("Early stopping")
                break

    if best_model is not None:
        model.load_state_dict(best_model)
    else:
        print(
            "Advertencia: ningún checkpoint superó val_loss inicial; "
            "se conserva el modelo del último epoch en lugar de un best_model."
        )
    return model, history
