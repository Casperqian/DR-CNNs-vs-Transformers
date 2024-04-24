# import cv2
import numpy as np
from PIL import Image
import torch
import torch
from torchvision.transforms import Compose, Lambda


def convert_rgba_to_rgb(pil_img):
    if len(pil_img.getbands()) == 4:  # Check if the image is RGBA
        r, g, b, a = pil_img.split()
        pil_img = Image.merge("RGB", (r, g, b))  # Merge only R, G, B channels
    return pil_img

class RGBA2RGB:
    def __init__(self):
        self.convert_rgba_to_rgb = convert_rgba_to_rgb

    def __call__(self, img):
        # Ensure the input is PIL Image
        if not isinstance(img, Image.Image):
            raise TypeError(f"Input type should be PIL Image. Got {type(img)}.")
        img = self.convert_rgba_to_rgb(img)

        return img

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.float().softmax(1) * x.float().log_softmax(1)).sum(1)