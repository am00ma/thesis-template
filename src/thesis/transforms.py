import torch
from torchvision.transforms import v2

TRANSFORM_DEFAULT = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
    ]
)


def POSTPROC_FLATTEN(x: torch.Tensor, _: tuple[torch.Tensor, ...]):
    """Convert [B, N, D] to [B, N*D]."""
    return x.view(x.size(0), -1)


def POSTPROC_BOXES(x: torch.Tensor, batch: tuple[torch.Tensor, ...]):
    """Stack bounding boxes with ref to idx of image."""
    idxs, *_ = batch
    y = []
    for idx, x_ in zip(idxs, x):
        y_ = torch.hstack([x_, torch.ones_like(x_[:, :1]) * idx])
        y.append(y_)
    y = torch.vstack(y)
    return y
