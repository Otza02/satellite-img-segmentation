import numpy as np
from pathlib import Path
from zipfile import ZipFile
import shutil
from torchvision import transforms
from PIL import Image
from satelliteSegmentation.tokenizer import Tokenizer
from tqdm import tqdm


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
        for file in tqdm(filter(lambda txt: not Path(txt).name.startswith("jp"), images_list)):
            with z.open(file, "r") as f_read, open(
                images_path / Path(file).name, "wb"
            ) as f_write:
                shutil.copyfileobj(f_read, f_write)
        print("Imagenes extraidas con exito")

        for file in tqdm(filter(lambda txt: not Path(txt).name.startswith("jp"), masks_list)):
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
    masks_id_path = Path(masks_id_path) if isinstance(masks_id_path, str) else masks_id_path
    masks_id_path.mkdir(parents=True, exist_ok=True)
    print("Convirtiendo mascaras color a mascaras id")
    to_tensor = transforms.PILToTensor()
    for mask_file in tqdm(masks_path.glob("*.png")):
        mask_id = Tokenizer.color2id(to_tensor(Image.open(mask_file)))
        Image.fromarray(mask_id.numpy().astype(np.uint8)).save(masks_id_path / mask_file.name)
    
    print(f"imagenes y mascaras guardadas en {masks_id_path.parent}")


def main():
    extract_zip("data/mumbai_raw.zip")
    masks_to_id()
    proc_folder = Path("data/processed")
    proc_folder.mkdir(exist_ok=True)
    shutil.move("data/raw/images", proc_folder)

if __name__ == "__main__":
    main()