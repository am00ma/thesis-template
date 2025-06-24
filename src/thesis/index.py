from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm

import distinctipy
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from thesis.config import Config
from thesis.paths import DATA_DIR
from thesis.utils.db import save_db


@dataclass()
class BoxesConfig:
    data_dir: Path = DATA_DIR
    root_dir: Path = field(init=False)

    name: str = "fixed-rectangles"

    num_images: int = 1024

    image_width: int = 640
    image_height: int = 480

    box_width: int = 40
    box_height: int = 20

    def __post_init__(self):
        self.root_dir = self.data_dir / self.name
        self.root_dir.mkdir(exist_ok=True, parents=True)


@dataclass()
class ClustersConfig:
    num_dims: int = 2
    num_clusters: int = 4
    num_points: int = 8 * 4

    seed: int = 7

    cluster_mean_min: float = -1.0
    cluster_mean_max: float = 1.0
    cluster_std_min: float = 0.1
    cluster_std_max: float = 0.3

    points_per_cluster_min: int = 2
    points_per_cluster_max: int = 16

    colors: list[tuple[int, int, int]] = field(init=False)

    def __post_init__(self):
        self.colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in distinctipy.get_colors(self.num_clusters, rng=self.seed)]


def normalize_points(points, w: int, h: int):
    # Get max distances
    max_x = np.max(np.abs(points[:, 0]))
    max_y = np.max(np.abs(points[:, 1]))

    # Scale the points
    points[:, 0] *= (w / 2) / max_x
    points[:, 1] *= (h / 2) / max_y

    # Center the points
    points[:, 0] += w / 2
    points[:, 1] += h / 2

    return points


def generate_clusters(cfg: ClustersConfig, exact_num_points: bool = False):
    # Set seed
    np.random.seed(cfg.seed)

    # Generate number of points per cluster
    nums = np.random.randint(low=cfg.points_per_cluster_min, high=cfg.points_per_cluster_max, size=cfg.num_clusters)

    # Algo to try to get enough points
    if exact_num_points:

        # 'Double' the sample for each cluster
        while nums.sum() < cfg.num_points:
            nums += np.random.randint(low=cfg.points_per_cluster_min, high=cfg.points_per_cluster_max, size=cfg.num_clusters)

        # Remove one sample from beginning till we reach desired num_points
        for i in range(nums.sum() - cfg.num_points):
            nums[i % cfg.num_clusters] -= 1

    # Generate each sample dim from normal dist independently
    points = []
    labels = []
    for i in range(cfg.num_clusters):
        points_ = np.empty((nums[i], cfg.num_dims), dtype=np.float32)
        labels_ = np.ones((nums[i],), dtype=int) * i
        for j in range(cfg.num_dims):
            mean = np.random.rand() * (cfg.cluster_mean_max - cfg.cluster_mean_min) + cfg.cluster_mean_min
            std = np.random.rand() * (cfg.cluster_std_max - cfg.cluster_std_min) + cfg.cluster_std_min
            points_[:, j] = np.random.normal(mean, std, nums[i])
        points.append(points_)
        labels.append(labels_)

    # Return points and cluster labels
    points = np.concat(points)  # (N, num_dims)
    labels = np.concat(labels)  # (N, 1)

    return points, labels


def index_boxes_dataset(cfg: Config):
    clusters_cfg = ClustersConfig(num_clusters=8, seed=cfg.data_seed)
    boxes_cfg = BoxesConfig(data_dir=cfg.data_dir)

    count = 0
    files = []
    boxes = []
    indices = []
    pbar = tqdm(range(boxes_cfg.num_images), total=boxes_cfg.num_images, desc="Generating")
    for i in pbar:
        clusters_cfg.seed = i
        points, labels = generate_clusters(clusters_cfg, exact_num_points=False)

        # normalize_points
        w, h = boxes_cfg.image_width, boxes_cfg.image_height
        points = normalize_points(points, w, h)

        # Track boxes as x1, y1, x2, y2, label
        x = points[:, 0]
        y = points[:, 1]
        bw, bh = boxes_cfg.box_width, boxes_cfg.box_height
        boxes_ = np.vstack([x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2, labels]).T
        boxes.append(boxes_)  # x1, y1, x2, y2, labels

        # Track indices (start and length indexed to images)
        indices.append((count, len(points)))
        count += len(points)

        # Plot image
        image = Image.new("RGB", (w, h), color="black")
        draw = ImageDraw.Draw(image)
        for _, ((x, y), l) in enumerate(zip(points, labels)):
            draw.rectangle((x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2), fill=clusters_cfg.colors[l])

        # Save image
        path = (boxes_cfg.root_dir / f"{i:05d}.jpg").relative_to(boxes_cfg.data_dir)
        image.save(boxes_cfg.data_dir / path)

        # Add to dataset dataframe
        row_file = {
            "idx": i,
            "path": str(path),
            "width": w,
            "height": h,
            "split": "train",
            "label": 0,
        }
        files.append(row_file)

    boxes = np.vstack(boxes)
    indices = np.vstack(indices)

    files_df = pd.DataFrame(files)
    boxes_df = pd.DataFrame(
        {
            "x1": boxes[:, 0],
            "y1": boxes[:, 1],
            "x2": boxes[:, 2],
            "y2": boxes[:, 3],
            "label": boxes[:, 4],
        }
    )
    indices_df = pd.DataFrame({"start": indices[:, 0], "length": indices[:, 1]})

    db_path = cfg.data_dir / f"{cfg.data_name}-{cfg.data_seed}.db"
    save_db(files_df, db_path, "dataset", verbose=True)
    save_db(boxes_df, db_path, "boxes", verbose=True)
    save_db(indices_df, db_path, "indices", verbose=True)


if __name__ == "__main__":
    from thesis.config import Config

    cfg = Config(data_name="fixed-rectangles", data_dir=DATA_DIR / "boxes", data_seed=7)

    index_boxes_dataset(cfg)
