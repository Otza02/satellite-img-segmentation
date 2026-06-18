from torch.utils.data.dataset import Dataset
from zipfile import ZipFile
from pathlib import Path
from typing import Literal
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import numpy as np
from PIL import Image


class SatelliteData(Dataset):
    def __init__(self, images_dir: str | Path, masks_dir: str | Path):
        images_dir = Path(images_dir) if isinstance(images_dir, str) else images_dir
        masks_dir = Path(masks_dir) if isinstance(masks_dir, str) else masks_dir
        X = []
        Y = []
        image_paths = sorted(images_dir.glob("*.tif"))
        for image_path in tqdm(image_paths):
            if not image_path.stem.startswith("tile"):
                continue

            mask_path = masks_dir / f"{image_path.stem}.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"No existe la máscara para {image_path.name}")

            image = np.array(Image.open(image_path))
            mask = np.array(Image.open(mask_path))

            image = torch.from_numpy(image).float()
            mask = torch.from_numpy(mask).float()

            X.append(image)
            Y.append(mask)

        self.X = torch.stack(X).permute(0, 3, 1, 2)
        self.Y = torch.stack(Y).unsqueeze(1)
        print(f"Dataset cargado:")
        print(f"X shape = {self.X.shape}")
        print(f"Y shape = {self.Y.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_data(
    folder: Literal["train", "val", "test"],
    base_path: str | Path = "data/processed",
):
    base = Path(base_path) if isinstance(base_path, str) else base_path
    path = base / str(folder)

    images = path / "images"
    if not Path.exists(images):
        raise Exception(f"No se encontró el directorio '{images}'")

    masks = path / "masks_id"
    if not Path.exists(masks):
        raise Exception(f"No se encontró el directorio '{masks}'")

    return SatelliteData(images, masks)


def main():
    load_data("train")


if __name__ == "__main__":
    main()
