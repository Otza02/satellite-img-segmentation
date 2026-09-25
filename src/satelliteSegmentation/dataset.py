from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
import torchvision.transforms as TF
from torchvision.transforms import v2
from torchvision import tv_tensors

from PIL import Image


class SatelliteData(Dataset):
    def __init__(
        self,
        data_dir: str | Path = "data/train",
        transform: v2.Compose | None = None,
        lazy: bool = False,
    ):
        self.tf = transform
        self.lazy = lazy
        data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.images_dir = data_dir / "images"
        self.masks_dir = data_dir / "masks"

        # Path.glob() no garantiza un orden estable entre sistemas de
        # archivos/entornos. Sin un orden determinista, la misma semilla del
        # generador usada para separar train/val puede terminar seleccionando
        # archivos distintos en cada corrida.
        self.file_stems = sorted(p.stem for p in self.images_dir.glob("*.tif"))

        if self.lazy:
            self.X = None
            self.Y = None
        else:
            to_tensor = TF.ToTensor()
            X = []
            Y = []
            for stem in tqdm(self.file_stems):
                img = to_tensor(Image.open(self.images_dir / f"{stem}.tif"))
                X.append(img)

                # decode_image devuelve uint8; CrossEntropyLoss necesita
                # targets de índice de clase en torch.long.
                msk = decode_image(self.masks_dir / f"{stem}.png").squeeze(0).long()  # type: ignore
                Y.append(msk)

            self.X = torch.stack(X)
            self.Y = torch.stack(Y)
            print(f"Dataset cargado:")
            print(f"X shape = {self.X.shape}")
            print(f"Y shape = {self.Y.shape}")

    def __len__(self):
        return len(self.file_stems)

    def _load(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        stem = self.file_stems[idx]
        to_tensor = TF.ToTensor()
        img = to_tensor(Image.open(self.images_dir / f"{stem}.tif"))
        msk = decode_image(self.masks_dir / f"{stem}.png").squeeze(0).long()  # type: ignore
        return img, msk

    def __getitem__(self, idx) -> tuple[tv_tensors.Image, tv_tensors.Mask]:
        if self.lazy:
            img, msk = self._load(idx)
        else:
            img, msk = self.X[idx], self.Y[idx]  # type: ignore

        img, msk = tv_tensors.Image(img), tv_tensors.Mask(msk)
        if self.tf is not None:
            return self.tf(img, msk)
        return img, msk


def spatial_train_val_split(
    dataset: SatelliteData,
    val_fraction: float = 0.2,
    n_blocks: int = 10,
    seed: int = 2026,
) -> tuple[list[int], list[int]]:
    """
    Separa los índices del dataset en train/val por bloques contiguos del
    orden (determinista, ver `file_stems`) en lugar de un split aleatorio por
    tile individual como hacía `random_split`.

    Los tiles satelitales vecinos en la escena de origen suelen quedar cerca
    entre sí una vez que los nombres de archivo están ordenados, así que un
    split puramente aleatorio por tile puede dejar tiles casi idénticos (o
    directamente solapados/adyacentes) tanto en train como en val, inflando
    de forma optimista la métrica de validación. Acá se arma la validación
    con bloques completos y contiguos (no tiles sueltos dispersos), lo que
    reduce cuántos pares train/val terminan siendo vecinos directos.
    """
    n = len(dataset)
    block_size = max(1, n // n_blocks)
    blocks = [list(range(i, min(i + block_size, n))) for i in range(0, n, block_size)]

    rng = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(blocks), generator=rng).tolist()

    target_val = int(round(n * val_fraction))
    val_idx: set[int] = set()
    for b in order:
        if len(val_idx) >= target_val:
            break
        val_idx.update(blocks[b])

    train_idx = [i for i in range(n) if i not in val_idx]
    val_idx_sorted = [i for i in range(n) if i in val_idx]
    return train_idx, val_idx_sorted


def main():
    from matplotlib import pyplot as plt
    from satelliteSegmentation.tokenizer import Tokenizer

    tf = v2.Compose(
        [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation([0, 180], fill=6),
        ]
    )

    data_folder = Path("data/final/train")
    data = SatelliteData(data_folder, transform=tf)
    x, y = data[0]

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x.permute(1, 2, 0))
    ax[1].imshow(Tokenizer.id2color(y).permute(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    main()
