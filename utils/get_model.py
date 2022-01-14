from .models import unet_2D


def get_model(cfg):
    if cfg.MODEL.NAME == 'unet':
        model = unet_2D(method=cfg.MODEL.METHOD, cfg=cfg)
    else:
        raise Exception("Model not found")

    return model
