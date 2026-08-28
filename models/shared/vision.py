"""Image-shape and layout helpers shared by visual model families."""

from collections.abc import Mapping

from .utils import parse_model_io


def image_spec(model_io):
    observations = (
        {str(key): tuple(map(int, shape)) for key, shape in model_io.items()}
        if isinstance(model_io, Mapping) and "observations" not in model_io
        else parse_model_io(model_io)[0]
    )
    images = [(key, shape) for key, shape in observations.items() if len(shape) == 3]
    if len(images) != 1:
        raise ValueError(f"Image models require one HWC observation, got {observations}.")
    key, shape = images[0]
    if shape[-1] != 3:
        raise ValueError(f"Expected an HWC RGB image, got {key}={shape}.")
    return key, shape


def channel_first(value):
    """Convert an HWC tensor with arbitrary leading dimensions to BCHW."""
    prefix = value.shape[:-3]
    value = value.reshape(-1, *value.shape[-3:]).permute(0, 3, 1, 2).float()
    return value, prefix
