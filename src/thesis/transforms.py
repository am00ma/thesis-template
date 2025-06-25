from typing import Any
import torch
from torchvision.transforms import v2
from torchvision.ops import nms
from torchvision.transforms.v2 import Transform

TRANSFORM_DEFAULT = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
    ]
)


class Flatten(Transform):
    """Convert [B, N, D] to [B, N*D]."""

    _transformed_types = (torch.Tensor,)

    def transform(self, inpt: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        return inpt.view(inpt.size(0), -1)


class NMSYoloBoxes(Transform):
    """Performs non-maximum-suppression on [32, 3087, 85] batch of yolo boxes.

    Uses torchvision.ops.nms with:
        conf_threshold = 0.3
        iou_threshold = 0.3

    Returns: [N, 86]
          Prefixes idx wrt batch to beggining of vector.
    """

    conf_threshold: float = 0.3
    iou_threshold: float = 0.3

    _transformed_types = (torch.Tensor,)

    def __init__(
        self,
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.3,
    ) -> None:
        super().__init__()

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def transform(self, inpt: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:

        filtered = []
        for idx, features in zip(range(len(inpt)), inpt):
            boxes = features[:, :4]  # xyhw
            scores = features[:, 4]

            # First threshold
            boxes = boxes[scores > self.conf_threshold, :]
            scores = scores[scores > self.conf_threshold]

            # Convert to x1, y1, x2, y2
            x1, y1, w, h = (boxes[:, i].unsqueeze(1) for i in range(4))
            x2, y2 = x1 + w, y1 + h
            boxes = torch.hstack([x1, y1, x2, y2])

            # Perform nms
            selected = nms(boxes, scores, iou_threshold=self.iou_threshold)

            # Append sample idx
            features = torch.hstack([features[selected], torch.ones_like(selected).unsqueeze(1) * idx])

            # Accumulate
            filtered.append(features)

        # Stack
        filtered = torch.vstack(filtered)

        return filtered


POSTPROC_YOLO_BOXES = v2.Compose(
    [
        NMSYoloBoxes(0.3, 0.3),
    ]
)
