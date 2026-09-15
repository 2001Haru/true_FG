"""Render a labeled contact sheet of SRe2L++ Aircraft IPC images."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--classes", nargs="+", type=int, default=(0, 19, 39, 59, 79, 99))
    a = p.parse_args()
    tile, gap, label_width, header, image_label = 224, 12, 150, 52, 24
    width = label_width + 3 * tile + 4 * gap
    height = header + len(a.classes) * (image_label + tile + gap) + gap
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    title_font, label_font = font(22), font(18)
    draw.text((12, 12), "Aircraft SRe2L++  |  Teacher42 / Recovery42 / IPC3", fill=(20, 20, 20), font=title_font)
    for row_index, class_id in enumerate(a.classes):
        class_dir = a.image_root / f"new{class_id:03d}"
        images = sorted(class_dir.glob("*.jpg"))
        if len(images) != 3:
            raise RuntimeError(f"expected three images in {class_dir}, got {len(images)}")
        row_y = header + row_index * (image_label + tile + gap)
        image_y = row_y + image_label
        draw.text((12, image_y + 90), f"class {class_id:03d}", fill=(20, 20, 20), font=label_font)
        for column, path in enumerate(images):
            with Image.open(path) as handle:
                image = handle.convert("RGB")
            if image.size != (tile, tile):
                raise RuntimeError(f"unexpected image size: {path} {image.size}")
            x = label_width + gap + column * (tile + gap)
            draw.text((x + 5, row_y + 3), path.stem.split("_")[-1], fill=(20, 20, 20), font=font(15))
            canvas.paste(image, (x, image_y))
            draw.rectangle((x, image_y, x + tile - 1, image_y + tile - 1), outline=(70, 70, 70), width=1)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.output.with_suffix(".png.tmp")
    canvas.save(temporary, format="PNG", compress_level=1)
    temporary.replace(a.output)
    print(a.output)


if __name__ == "__main__":
    main()
