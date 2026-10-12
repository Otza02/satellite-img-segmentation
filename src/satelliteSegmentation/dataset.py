from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as TF
from torchvision.io import decode_image
from torchvision.transforms import v2
from torchvision import tv_tensors

from PIL import Image


class SatelliteData(Dataset):
    def __init__(
        self,
        data_dir: str | Path = "data/train",
        lazy: bool = False,
    ):
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
        return img, msk


class AugmentedSubset(Dataset):
    """
    Envuelve un subconjunto de índices de un `SatelliteData` con su propio
    `transform`.
    """

    def __init__(
        self,
        dataset: SatelliteData,
        indices: list[int],
        transform: v2.Compose | None = None,
        color_transform: v2.Compose | None = None,
    ):
        self.dataset = dataset
        self.indices = indices
        self.tf = transform
        self.color_tf = color_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx) -> tuple[tv_tensors.Image, tv_tensors.Mask]:
        real_idx = self.indices[idx]
        img, msk = self.dataset[real_idx]

        img, msk = tv_tensors.Image(img), tv_tensors.Mask(msk)
        if self.tf is not None:
            img_tf, msk_tf = self.tf(img, msk)
            if self.color_tf is not None:
                return self.color_tf(img_tf), msk_tf
            return img_tf, msk_tf
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


def spatial_kfold_indices(
    n: int,
    k: int = 5,
    n_blocks: int = 20,
) -> list[tuple[list[int], list[int]]]:
    """
    Devuelve k pares (train_idx, val_idx) para cross validation.

    Igual que `spatial_train_val_split`, trabaja con bloques contiguos del orden
    (determinista) de archivos en vez de tiles sueltos, para reducir la fuga
    espacial entre train y val. Los bloques se mezclan con `seed` y se reparten
    entre los k folds, de modo que cada tile aparece en validación exactamente
    una vez y los folds quedan de tamaño parecido.
    """
    if k < 2:
        raise ValueError("k debe ser >= 2")
    n_blocks = min(n_blocks, n)
    # n_blocks bloques contiguos de tamaño casi igual (difieren a lo sumo en 1)
    blocks = [
        list(range(b * n // n_blocks, (b + 1) * n // n_blocks)) for b in range(n_blocks)
    ]
    if len(blocks) < k:
        raise ValueError(
            f"Hay {len(blocks)} bloques y se pidieron {k} folds: subí n_blocks o bajá k"
        )

    order = list(range(len(blocks)))

    folds = []
    for f in range(k):
        val_blocks = set(order[f::k])
        val_idx = [i for b in sorted(val_blocks) for i in blocks[b]]
        val_set = set(val_idx)
        train_idx = [i for i in range(n) if i not in val_set]
        folds.append((train_idx, val_idx))
    return folds


def kfold_loaders(
    dataset: SatelliteData,
    k: int = 5,
    batch_size: int = 32,
    shape_transform: v2.Compose | None = None,
    color_transform: v2.Compose | None = None,
    n_blocks: int = 20,
    **loader_kwargs,
):
    """
    Generador que rinde `(fold, train_loader, val_loader)` para cada uno de los k
    folds, reutilizando SIEMPRE la misma instancia de `SatelliteData`.

        for fold, train_loader, val_loader in kfold_loaders(data, k=5):
            ...

    `shape_transform` se aplica solo al loader de train. `loader_kwargs` se pasa
    a ambos DataLoader (num_workers, pin_memory, etc.).
    """
    for fold, (train_idx, val_idx) in enumerate(
        spatial_kfold_indices(len(dataset), k, n_blocks)
    ):
        train = AugmentedSubset(dataset, train_idx, shape_transform, color_transform)
        val = AugmentedSubset(dataset, val_idx)
        train_loader = DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(val, batch_size=batch_size, **loader_kwargs)
        yield fold, train_loader, val_loader


def main():
    from matplotlib import pyplot as plt
    from satelliteSegmentation.tokenizer import Tokenizer

    train_tf = v2.Compose(
        [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation([0, 180], fill=6),
        ]
    )

    data_folder = Path("data/final/train")
    data = SatelliteData(data_folder)

    train_idx, val_idx = spatial_train_val_split(data)
    train = AugmentedSubset(data, train_idx, transform=train_tf)
    val = AugmentedSubset(data, val_idx, transform=None)

    x, y = train[0]

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x.permute(1, 2, 0))
    ax[1].imshow(Tokenizer.id2color(y).permute(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    main()
