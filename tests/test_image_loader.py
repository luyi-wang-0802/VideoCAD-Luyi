from pathlib import Path

import torch
from PIL import Image

from data_loader.image_loader import ScreenshotImageLoader


def test_screenshot_image_loader_can_return_float16(tmp_path: Path) -> None:
    image_path = tmp_path / "screen.png"
    Image.new("L", (8, 4), color=128).save(image_path)
    loader = ScreenshotImageLoader(repo_root=tmp_path, image_size=(8, 4), dtype=torch.float16)

    image, available = loader.load(image_path)

    assert available
    assert image.dtype == torch.float16
