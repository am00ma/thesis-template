import torch
from thesis.config import Config


def load_model(cfg: Config) -> torch.nn.Module:

    if "yolo" in cfg.model_name:

        model = torch.hub.load(
            "ultralytics/yolov5",
            cfg.model_name,
            pretrained=cfg.model_yolo_pretrained,
        )

    elif "unet" in cfg.model_name:

        model = torch.hub.load(
            "mateuszbuda/brain-segmentation-pytorch",
            cfg.model_name,
            in_channels=cfg.model_unet_in_channels,
            out_channels=cfg.model_unet_out_channels,
            init_features=cfg.model_unet_init_features,
            pretrained=cfg.model_unet_pretrained,
        )

    else:

        raise KeyError(f"Invalid model_name: {cfg.model_name}")

    assert isinstance(model, torch.nn.Module)
    return model


if __name__ == "__main__":
    cfg = Config(model_name="yolov5s")
    model = load_model(cfg)
    print(model)

    cfg = Config(model_name="unet")
    model = load_model(cfg)
    print(model)
