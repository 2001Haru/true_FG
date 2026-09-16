"""Create paired contact sheets for the three NRR-inspired image optimization arms."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFont, ImageOps


ARMS = ("plain", "cam_protected", "random_protected")
LABELS = {
    "plain": "Plain optimization",
    "cam_protected": "CAM protected",
    "random_protected": "Random protected",
}


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for path in (Path("/usr/share/fonts/truetype/dejavu") / name, Path(name)):
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            pass
    return ImageFont.load_default()


def load_rgb(path):
    return Image.open(path).convert("RGB")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--r0", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--classes", default="000,014,032,048,068,094")
    p.add_argument("--slot", type=int, default=0)
    a = p.parse_args()
    classes = a.classes.split(",")
    manifests = {}
    for arm in ARMS:
        data = json.loads((a.root / f"construction/arm_manifests/{arm}.json").read_text())
        manifests[arm] = {(row["class_folder"], row["slot"]): row for row in data["exports"]}

    rows = []
    for cls in classes:
        base = manifests["plain"][(cls, a.slot)]
        identity = base["identity"]
        entries = [manifests[arm][(cls, a.slot)] for arm in ARMS]
        if any(entry["identity"] != identity for entry in entries):
            raise RuntimeError(f"identity mismatch for class {cls}, slot {a.slot}")
        original = a.r0 / cls / identity
        if not original.exists():
            raise FileNotFoundError(original)
        rows.append((cls, identity, original, entries))

    tile, gap, label_h, header_h, left = 224, 14, 40, 54, 172
    cols = 4
    width = left + cols * tile + (cols - 1) * gap + 18
    height = header_h + len(rows) * (tile + label_h + gap) + 8
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    head_font, row_font, small_font = font(22, True), font(18, True), font(14)
    headers = ("R0 original",) + tuple(LABELS[arm] for arm in ARMS)
    for col, text in enumerate(headers):
        x = left + col * (tile + gap)
        draw.text((x, 13), text, fill="black", font=head_font)
    for ridx, (cls, identity, original_path, entries) in enumerate(rows):
        y = header_h + ridx * (tile + label_h + gap)
        draw.text((12, y + 82), f"class {cls}\n{identity}\nslot {a.slot}", fill="black", font=row_font, spacing=6)
        original = load_rgb(original_path)
        images = [original] + [load_rgb(entry["path"]) for entry in entries]
        for col, image in enumerate(images):
            x = left + col * (tile + gap)
            sheet.paste(image, (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(128, 128, 128), width=1)
            if col:
                entry = entries[col - 1]
                note = f"mean |Δ|={entry['mean_abs_change']:.4f}"
                draw.text((x, y + tile + 7), note, fill="black", font=small_font)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = a.output_dir / "nrr_three_arm_contact_sheet.png"
    sheet.save(sheet_path)

    diff_cols = 3
    diff_width = left + diff_cols * tile + (diff_cols - 1) * gap + 18
    diff = Image.new("RGB", (diff_width, height), "white")
    ddraw = ImageDraw.Draw(diff)
    for col, arm in enumerate(ARMS):
        x = left + col * (tile + gap)
        ddraw.text((x, 13), f"{LABELS[arm]} |Δ| ×4", fill="black", font=head_font)
    for ridx, (cls, identity, original_path, entries) in enumerate(rows):
        y = header_h + ridx * (tile + label_h + gap)
        ddraw.text((12, y + 82), f"class {cls}\n{identity}\nslot {a.slot}", fill="black", font=row_font, spacing=6)
        original = load_rgb(original_path)
        for col, entry in enumerate(entries):
            x = left + col * (tile + gap)
            delta = ImageChops.difference(load_rgb(entry["path"]), original)
            delta = ImageEnhance.Brightness(delta).enhance(4.0)
            diff.paste(delta, (x, y))
            ddraw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(128, 128, 128), width=1)
            ddraw.text((x, y + tile + 7), f"actual mean |Δ|={entry['mean_abs_change']:.4f}", fill="black", font=small_font)
    diff_path = a.output_dir / "nrr_three_arm_difference_x4.png"
    diff.save(diff_path)
    print(json.dumps({"contact_sheet": str(sheet_path), "difference_sheet": str(diff_path),
                      "classes": classes, "slot": a.slot}, indent=2))


if __name__ == "__main__":
    main()
