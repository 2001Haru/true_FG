"""Unit check for per-file full-frame override during FKD metadata replay."""
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relabel.utils_fkd import ComposeWithCoords, ImageFolder_FKD_MIX, RandomResizedCropWithCoords


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "images/000"
        root.mkdir(parents=True)
        for name in ("slot0_full.png", "slot1_mosaic.png", "slot2_mosaic.png"):
            Image.new("RGB", (224, 224), (20, 30, 40)).save(root / name)
        transform = ComposeWithCoords(transforms=[RandomResizedCropWithCoords(
            size=224, scale=(0.08, 1.0), interpolation=InterpolationMode.BILINEAR)])
        dataset = ImageFolder_FKD_MIX(
            fkd_path=str(Path(directory) / "unused"), mode="fkd_save", root=str(root.parent),
            transform=transform, force_full_prefix="slot0_full")
        dataset.mode = "fkd_load"
        source_coords = torch.tensor([[0.2, 0.2, 0.5, 0.5]] * 3)
        dataset.batch_config = [source_coords, torch.tensor([False] * 3), None]
        dataset.batch_config_idx = 0
        returned = [dataset[index][3] for index in range(3)]
        assert torch.equal(returned[0], torch.tensor([0.0, 0.0, 1.0, 1.0]))
        assert torch.equal(returned[1], source_coords[1]) and torch.equal(returned[2], source_coords[2])
    print("FKD force-full-prefix test passed")


if __name__ == "__main__":
    main()
