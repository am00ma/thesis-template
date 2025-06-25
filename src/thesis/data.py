from typing import override

from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from thesis.paths import DATA_DIR
from thesis.utils.db import load_db
from thesis.transforms import TRANSFORM_DEFAULT


# idx, image, label, path
ItemType = tuple[int, torch.Tensor, int, str]


class ImageClassification(Dataset[ItemType]):
    # Inputs
    data_dir: Path
    name: str
    split: str
    seed: int

    # Attributes
    df: pd.DataFrame

    def __init__(
        self,
        data_dir: Path,
        name: str,
        split: str,
        seed: int,
        transform: v2.Transform = TRANSFORM_DEFAULT,
    ):
        self.name = name
        self.split = split
        self.seed = seed
        self.data_dir = data_dir
        self.db_path = data_dir / f"{self.name}-{self.seed}.db"

        if not self.db_path.exists():
            raise FileNotFoundError(f"DB file not found: {self.db_path}")

        df = load_db(self.db_path, "dataset")
        df = df[df["split"] == split]

        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    @override
    def __getitem__(self, idx) -> ItemType:
        row = self.df.iloc[idx]

        idx = int(row.idx)
        label = int(row.label)
        path = str(self.data_dir / row.path)

        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        return idx, image, label, path


if __name__ == "__main__":
    from thesis.config import Config

    cfg = Config(
        data_dir=DATA_DIR / "boxes",
        data_name="fixed-rectangles",
        data_split="train",
        data_seed=7,
    )

    dataset = ImageClassification(
        data_dir=cfg.data_dir,
        name=cfg.data_name,
        split=cfg.data_split,
        seed=cfg.data_seed,
    )

    print(f"{len(dataset)} samples\n")
    for i in range(5):
        sample = dataset[i]
        print(f"{sample[0]} (idx)")
        print(f"    image: {sample[1].shape}")
        print(f"    label: {sample[2]}")
        print(f"    path : {sample[3]}")
