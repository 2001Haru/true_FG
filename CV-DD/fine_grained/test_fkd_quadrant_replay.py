"""Unit checks for balanced, replayable single-quadrant FKD transforms."""

import tempfile
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relabel.utils_fkd import (
    ComposeWithCoords,
    ImageFolder_FKD_MIX,
    RandomHorizontalFlipWithRes,
    RandomResizedCropWithCoords,
    SelectQuadrantWithRes,
)


def main() -> None:
    colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
    image = Image.new("RGB", (224, 224))
    for quadrant, color in enumerate(colors):
        row, column = divmod(quadrant, 2)
        image.paste(Image.new("RGB", (112, 112), color), (column * 112, row * 112))
    selector = SelectQuadrantWithRes()
    for quadrant, color in enumerate(colors):
        selected, returned = selector(image, quadrant)
        assert selected.size == (112, 112)
        assert selected.getpixel((56, 56)) == color
        assert returned == quadrant

    transform = ComposeWithCoords(
        transforms=[
            SelectQuadrantWithRes(),
            RandomResizedCropWithCoords(
                size=224, scale=(0.5, 1.0), interpolation=InterpolationMode.BILINEAR
            ),
            RandomHorizontalFlipWithRes(),
        ]
    )
    full_crop = torch.tensor([0.0, 0.0, 1.0, 1.0])
    transformed, flip, coords, quadrant = transform(image, full_crop, False, 2)
    assert transformed.size == (224, 224)
    assert transformed.getpixel((112, 112)) == colors[2]
    assert flip is False and quadrant == 2 and torch.equal(coords, full_crop)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "images/00000"
        root.mkdir(parents=True)
        for index in range(300):
            Image.new("RGB", (2, 2), (index % 256, 0, 0)).save(root / f"{index:05d}.png")
        dataset = ImageFolder_FKD_MIX(
            fkd_path=str(Path(directory) / "fkd"),
            mode="fkd_save",
            quadrant_mode=True,
            quadrant_seed=42,
            root=str(root.parent),
            transform=None,
        )
        per_image = [[] for _ in range(len(dataset))]
        for epoch in range(4):
            dataset.set_epoch(epoch)
            values = []
            for index in range(len(dataset)):
                *_, quadrant = dataset[index]
                values.append(quadrant)
                per_image[index].append(quadrant)
            assert Counter(values) == Counter({0: 75, 1: 75, 2: 75, 3: 75})
        assert all(sorted(values) == [0, 1, 2, 3] for values in per_image)

        replay_transform = ComposeWithCoords(
            transforms=[
                SelectQuadrantWithRes(),
                RandomResizedCropWithCoords(
                    size=224, scale=(0.5, 1.0), interpolation=InterpolationMode.BILINEAR
                ),
                RandomHorizontalFlipWithRes(),
                transforms.ToTensor(),
            ]
        )
        save_dataset = ImageFolder_FKD_MIX(
            fkd_path=str(Path(directory) / "fkd"), mode="fkd_save",
            quadrant_mode=True, quadrant_seed=42, root=str(root.parent),
            transform=replay_transform,
        )
        save_dataset.set_epoch(7)
        order = [3, 1, 2, 0]
        torch.manual_seed(123)
        saved = [save_dataset[index] for index in order]
        images = torch.stack([row[0] for row in saved])
        flips = torch.tensor([row[2] for row in saved])
        coords = torch.stack([row[3] for row in saved])
        quadrants = torch.tensor([row[4] for row in saved])
        load_dataset = ImageFolder_FKD_MIX(
            fkd_path=str(Path(directory) / "unused"), mode="fkd_save",
            quadrant_mode=False, root=str(root.parent), transform=replay_transform,
        )
        load_dataset.mode = "fkd_load"
        load_dataset.batch_config = [coords, flips, quadrants]
        load_dataset.batch_config_idx = 0
        replayed = torch.stack([load_dataset[index][0] for index in order])
        assert torch.equal(images, replayed)
    print("FKD quadrant replay tests passed")


if __name__ == "__main__":
    main()
