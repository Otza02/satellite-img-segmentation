from satelliteSegmentation.tokenizer import Tokenizer
from PIL import Image
from pathlib import Path
import numpy as np
from torchvision.io import decode_image


def color_to_class(masks_folder: Path, masks_id_folder: Path):
    masks_id_folder.mkdir(parents=True, exist_ok=True)

    for msk in masks_folder.glob("*.png"):
        img = decode_image(msk.as_posix())
        img_id = Tokenizer.color2id(img)
        img_path = masks_id_folder / f"{msk.stem}.png"
        Image.fromarray(img_id.cpu().numpy().astype(np.uint8)).save(img_path)


def main():
    masks_folder = Path.cwd() / "data" / "processed" / "test" / "masks"
    masks_id_folder = Path.cwd() / "data" / "processed" / "test" / "masks_id"
    color_to_class(masks_folder, masks_id_folder)
    print("Procesado finalizado para test")

    masks_folder = Path.cwd() / "data" / "processed" / "train" / "masks"
    masks_id_folder = Path.cwd() / "data" / "processed" / "train" / "masks_id"
    color_to_class(masks_folder, masks_id_folder)
    print("Procesado finalizado para train")

    masks_folder = Path.cwd() / "data" / "processed" / "val" / "masks"
    masks_id_folder = Path.cwd() / "data" / "processed" / "val" / "masks_id"
    color_to_class(masks_folder, masks_id_folder)
    print("Procesado finalizado para val")


if __name__ == "__main__":
    main()
