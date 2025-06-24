from dataclasses import dataclass
from pathlib import Path
from thesis.paths import DATA_DIR, OUTPUT_DIR, CONFIG_DIR


@dataclass()
class Config:
    config_path: Path = CONFIG_DIR / "default.json"

    # Basics
    debug: bool = False
    data_dir: Path = DATA_DIR
    output_dir: Path = OUTPUT_DIR

    # Data
    data_name: str = "boxes"
    data_split: str = "train"
    data_frac: float = 1.0
    data_seed: int = 7

    # Model - common
    model_name: str = "yolov5s"

    # --- Individual model configs ---

    # yolo
    model_yolo_pretrained: bool = True

    # unet
    model_unet_in_channels: int = 3
    model_unet_out_channels: int = 1
    model_unet_init_features: int = 32
    model_unet_pretrained: bool = True
