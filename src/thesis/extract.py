from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from thesis.config import Config
from thesis.data import ImageClassification
from thesis.models import load_model
from thesis.paths import DATA_DIR
from thesis.transforms import NMSYoloBoxes

postproc_nms = NMSYoloBoxes(0.3, 0.3)


def extract_features(cfg: Config, device: torch.device):
    """Saves features to npz file with key 'data'."""

    dataset = ImageClassification(
        data_dir=cfg.data_dir,
        name=cfg.data_name,
        split=cfg.data_split,
        seed=cfg.data_seed,
    )
    dataloader = DataLoader(dataset, shuffle=False, batch_size=cfg.data_batch_size)
    model = load_model(cfg, device, train=False)

    data = []
    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader), desc="Extracting")
        for batch in pbar:
            # idx, image, label, path
            _, images, _, _ = batch

            images = images.to(device)
            out = model(images)

            out = postproc_nms(out)

            data.append(out)

    data = torch.cat(data, dim=0).cpu().numpy()
    print(data.shape)


if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = Config(
        data_dir=DATA_DIR / "boxes",
        data_name="fixed-rectangles",
        data_split="train",
        data_seed=7,
        model_name="yolov5s",
        model_yolo_pretrained=True,
    )

    dataset = ImageClassification(
        data_dir=cfg.data_dir,
        name=cfg.data_name,
        split=cfg.data_split,
        seed=cfg.data_seed,
    )

    extract_features(cfg, DEVICE)
