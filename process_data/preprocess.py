import numpy as np
import pandas as pd
from pathlib import Path
from zipfile import ZipFile
import shutil
from torchvision import transforms
from torchvision.io import decode_image
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from satelliteSegmentation.tokenizer import Tokenizer


def extract_zip(zip_path: str, raw_path: str | Path = "data/raw"):
    raw_path = Path(raw_path) if isinstance(raw_path, str) else raw_path
    with ZipFile(zip_path) as z:
        prepared = filter(lambda txt: "Prepared_Dataset" in str(txt), z.namelist())
        images_list = filter(lambda txt: "images" in str(txt), prepared)

        prepared = filter(lambda txt: "Prepared_Dataset" in str(txt), z.namelist())
        masks_list = filter(lambda txt: "masks" in str(txt), prepared)

        images_path = raw_path / "images"
        images_path.mkdir(exist_ok=True, parents=True)

        masks_path = raw_path / "masks"
        masks_path.mkdir(exist_ok=True, parents=True)

        print("Iniciando extraccion de zip")
        for file in tqdm(
            filter(lambda txt: not Path(txt).name.startswith("jp"), images_list)
        ):
            with z.open(file, "r") as f_read, open(
                images_path / Path(file).name, "wb"
            ) as f_write:
                shutil.copyfileobj(f_read, f_write)
        print("Imagenes extraidas con exito")

        for file in tqdm(
            filter(lambda txt: not Path(txt).name.startswith("jp"), masks_list)
        ):
            with z.open(file, "r") as f_read, open(
                masks_path / Path(file).name, "wb"
            ) as f_write:
                shutil.copyfileobj(f_read, f_write)
        print("Mascaras extraidas con exito")


def masks_to_id(
    masks_path: str | Path = "data/raw/masks",
    masks_id_path: str | Path = "data/processed/masks_id",
):
    masks_path = Path(masks_path) if isinstance(masks_path, str) else masks_path
    masks_id_path = (
        Path(masks_id_path) if isinstance(masks_id_path, str) else masks_id_path
    )
    masks_id_path.mkdir(parents=True, exist_ok=True)
    print("Convirtiendo mascaras color a mascaras id")
    to_tensor = transforms.PILToTensor()
    for mask_file in tqdm(masks_path.glob("*.png")):
        mask_id = Tokenizer.color2id(to_tensor(Image.open(mask_file)))
        Image.fromarray(mask_id.numpy().astype(np.uint8)).save(
            masks_id_path / mask_file.name
        )

    print(f"imagenes y mascaras guardadas en {masks_id_path.parent}")


def _frecuencies(masks_id: Path):
    histograms = []
    images = []
    # Conteo de frecuencias
    for image_path in masks_id.glob("*.png"):
        mask = decode_image(image_path) # type: ignore

        hist = torch.bincount(mask.flatten(), minlength=7).float()

        hist /= hist.sum()

        histograms.append(hist)
        images.append(image_path)

    df = pd.DataFrame(
        torch.stack(histograms).numpy(), columns=[f"class_{i}" for i in range(7)]
    )

    df["image"] = images
    df["class5_bin"] = pd.qcut(df["class_5"], q=5, duplicates="drop")
    return df


def _compute_distribution(paths):
    counts = torch.zeros(7)

    for path in paths:
        mask = decode_image(path)

        counts += torch.bincount(mask.flatten(), minlength=7)

    return counts / counts.sum()


def final_zip(
    proc_path: str | Path = "data/processed",
    final_path: str | Path = "data/final",
):
    proc_path = Path(proc_path) if isinstance(proc_path, str) else proc_path
    final_path = Path(final_path) if isinstance(final_path, str) else final_path

    df = _frecuencies(proc_path / "masks_id")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=2026, stratify=df["class5_bin"]
    )
    print("Final distributions")
    print(f" Distribucion train: {_compute_distribution(train_df["image"].values)}")
    print(f" Distribucion test: {_compute_distribution(test_df["image"].values)}")

    train_images_dir = final_path / "train" / "images"
    train_images_dir.mkdir(exist_ok=True, parents=True)
    
    train_masks_dir = final_path / "train" / "masks"
    train_masks_dir.mkdir(exist_ok=True, parents=True)
    print("Moviendo imagenes a final/train")
    for msk in tqdm(train_df["image"].values):
        img_name = msk.name.replace(".png", ".tif")
        msk_name = msk.name
        shutil.move(proc_path / "images" / img_name, train_images_dir / img_name)
        shutil.move(proc_path / "masks_id" / msk_name, train_masks_dir / msk_name)
    
    test_images_dir = final_path / "test" / "images"
    test_images_dir.mkdir(exist_ok=True, parents=True)
    
    test_masks_dir = final_path / "test" / "masks"
    test_masks_dir.mkdir(exist_ok=True, parents=True)
    print("Moviendo imagenes a final/test")
    for msk in tqdm(test_df["image"].values):
        img_name = msk.name.replace(".png", ".tif")
        msk_name = msk.name
        shutil.move(proc_path / "images" / img_name, test_images_dir / img_name)
        shutil.move(proc_path / "masks_id" / msk_name, test_masks_dir / msk_name)
    
    shutil.make_archive(final_path / "train", "zip", final_path) # type: ignore
    shutil.make_archive(final_path / "test", "zip", final_path) # type: ignore

def main():
    extract_zip("data/mumbai_raw.zip")
    masks_to_id()
    proc_folder = Path("data/processed")
    proc_folder.mkdir(exist_ok=True)
    shutil.move("data/raw/images", proc_folder)
    
    final_zip(proc_folder, "data/final")

if __name__ == "__main__":
    main()
