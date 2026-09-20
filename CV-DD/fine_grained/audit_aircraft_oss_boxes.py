"""Evaluate Grounding DINO OSS and Grounding DINO + SAM 2.1 on Aircraft boxes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clip_box(box, width, height):
    x1, y1, x2, y2 = map(float, box)
    x1, x2 = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
    y1, y2 = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
    return [x1, y1, x2, y2]


def box_metrics(prediction, target, width, height):
    p = clip_box(prediction, width, height); t = clip_box(target, width, height)
    ix1, iy1 = max(p[0], t[0]), max(p[1], t[1])
    ix2, iy2 = min(p[2], t[2]), min(p[3], t[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    pa = max(0.0, p[2] - p[0]) * max(0.0, p[3] - p[1])
    ta = max(0.0, t[2] - t[0]) * max(0.0, t[3] - t[1])
    union = pa + ta - intersection
    pc = ((p[0] + p[2]) / 2, (p[1] + p[3]) / 2)
    tc = ((t[0] + t[2]) / 2, (t[1] + t[3]) / 2)
    return {
        "iou": intersection / union if union else 0.0,
        "official_coverage": intersection / ta if ta else 0.0,
        "predicted_precision": intersection / pa if pa else 0.0,
        "area_ratio": pa / ta if ta else float("nan"),
        "abs_log_area_ratio": abs(math.log(max(pa, 1e-12) / max(ta, 1e-12))),
        "center_error_diagonal": math.hypot(pc[0] - tc[0], pc[1] - tc[1]) / math.hypot(width, height),
    }


def mask_box(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs): return None
    # Continuous XYXY convention: the last included pixel ends at index+1.
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def distribution(values):
    values = np.asarray(list(values), dtype=np.float64)
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)),
            "min": float(values.min()), "p10": float(np.quantile(values, .1)),
            "median": float(np.median(values)), "p90": float(np.quantile(values, .9)),
            "max": float(values.max())}


def summarize(rows, method):
    valid = [row for row in rows if row[method]["box_xyxy"] is not None]
    result = {"images": len(rows), "valid": len(valid), "missing": len(rows) - len(valid)}
    for metric in ("iou", "official_coverage", "predicted_precision", "area_ratio",
                   "abs_log_area_ratio", "center_error_diagonal"):
        result[metric] = distribution(row[method]["metrics"][metric] for row in valid)
    result["recall_iou_50"] = float(np.mean([row[method]["metrics"]["iou"] >= .5 for row in rows]))
    result["recall_iou_75"] = float(np.mean([row[method]["metrics"]["iou"] >= .75 for row in rows]))
    result["recall_iou_90"] = float(np.mean([row[method]["metrics"]["iou"] >= .9 for row in rows]))
    by_variant = defaultdict(list)
    for row in valid: by_variant[row["variant"]].append(row[method]["metrics"]["iou"])
    result["variant_macro_iou"] = float(np.mean([np.mean(values) for values in by_variant.values()]))
    result["variants"] = len(by_variant)
    return result


def aircraft_rows(raw_images, boxes_path, variants_path):
    boxes = {parts[0]: list(map(float, parts[1:5]))
             for line in boxes_path.read_text().splitlines() if (parts := line.split())}
    variants = {}
    for line in variants_path.read_text().splitlines():
        identity, variant = line.split(" ", 1); variants[identity] = variant
    if set(variants) - set(boxes): raise RuntimeError("variant identities missing official boxes")
    rows = []
    for identity in sorted(variants):
        path = raw_images / f"{identity}.jpg"
        if not path.is_file(): raise FileNotFoundError(path)
        with Image.open(path) as image: width, height = image.size
        x1, y1, x2, y2 = boxes[identity]
        rows.append({"identity": identity, "variant": variants[identity], "path": path,
                     "width": width, "height": height,
                     "official_box_xyxy": [x1 - 1, y1 - 1, x2, y2]})
    return rows


class GroundingDinoTopOne:
    def __init__(self, source_root, config, checkpoint, device, caption, text_encoder_root=None):
        sys.path.insert(0, str(source_root.resolve()))
        import groundingdino.datasets.transforms as T
        from groundingdino.models import build_model
        from groundingdino.util.misc import clean_state_dict
        from groundingdino.util.slconfig import SLConfig
        args = SLConfig.fromfile(str(config)); args.device = device
        if text_encoder_root is not None: args.text_encoder_type = str(text_encoder_root.resolve())
        self.model = build_model(args)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = self.model.load_state_dict(clean_state_dict(payload["model"]), strict=False)
        self.load_audit = {"missing": list(missing), "unexpected": list(unexpected)}
        self.model = self.model.to(device).eval(); self.device = device
        self.caption = caption.lower().strip()
        if not self.caption.endswith("."): self.caption += "."
        self.transform = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(),
                                    T.Normalize([.485, .456, .406], [.229, .224, .225])])

    @torch.inference_mode()
    def __call__(self, image):
        transformed, _ = self.transform(image, None)
        output = self.model(transformed[None].to(self.device), captions=[self.caption])
        logits = output["pred_logits"].sigmoid()[0]
        scores = logits.max(1).values; index = int(scores.argmax())
        cx, cy, width, height = output["pred_boxes"][0, index].float().cpu().tolist()
        iw, ih = image.size
        box = [(cx - width / 2) * iw, (cy - height / 2) * ih,
               (cx + width / 2) * iw, (cy + height / 2) * ih]
        return clip_box(box, iw, ih), float(scores[index]), index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-images", required=True, type=Path)
    parser.add_argument("--boxes", required=True, type=Path)
    parser.add_argument("--variants", required=True, type=Path)
    parser.add_argument("--grounding-root", required=True, type=Path)
    parser.add_argument("--grounding-config", required=True, type=Path)
    parser.add_argument("--grounding-checkpoint", required=True, type=Path)
    parser.add_argument("--text-encoder-root", type=Path)
    parser.add_argument("--sam-root", required=True, type=Path)
    parser.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam-checkpoint", required=True, type=Path)
    parser.add_argument("--caption", default="airplane.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--disable-sam", action="store_true")
    args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    inputs = aircraft_rows(args.raw_images, args.boxes, args.variants)
    if args.limit is not None: inputs = inputs[:args.limit]
    dino = GroundingDinoTopOne(args.grounding_root, args.grounding_config,
                               args.grounding_checkpoint, args.device, args.caption,
                               args.text_encoder_root)
    sam = None
    if not args.disable_sam:
        sys.path.insert(0, str(args.sam_root.resolve()))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam = SAM2ImagePredictor(build_sam2(args.sam_config, str(args.sam_checkpoint),
                                            device=args.device, mode="eval"))
    output_jsonl = args.output_root / "predictions.jsonl"
    existing = {}
    if output_jsonl.is_file():
        for line in output_jsonl.read_text().splitlines():
            row = json.loads(line); existing[row["identity"]] = row
    with output_jsonl.open("a", buffering=1) as writer:
        for number, item in enumerate(inputs, 1):
            if item["identity"] in existing: continue
            with Image.open(item["path"]) as handle: image = handle.convert("RGB")
            dino_box, dino_score, query_index = dino(image)
            dino_result = {"box_xyxy": dino_box, "score": dino_score,
                           "query_index": query_index,
                           "metrics": box_metrics(dino_box, item["official_box_xyxy"], item["width"], item["height"])}
            sam_result = {"box_xyxy": None, "predicted_iou": None,
                          "mask_area_fraction": None,
                          "metrics": {key: 0.0 for key in dino_result["metrics"]}}
            if sam is not None:
                image_array = np.array(image, copy=True)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                             enabled=args.device.startswith("cuda")):
                    sam.set_image(image_array)
                    masks, scores, _ = sam.predict(box=np.asarray(dino_box, dtype=np.float32),
                                                   multimask_output=False)
                selected = int(np.argmax(scores)); box = mask_box(masks[selected])
                if box is not None:
                    sam_result = {"box_xyxy": clip_box(box, item["width"], item["height"]),
                                  "predicted_iou": float(scores[selected]),
                                  "mask_area_fraction": float(np.mean(masks[selected])),
                                  "metrics": box_metrics(box, item["official_box_xyxy"], item["width"], item["height"])}
            row = {"identity": item["identity"], "variant": item["variant"],
                   "raw_size": [item["width"], item["height"]],
                   "official_box_xyxy": item["official_box_xyxy"],
                   "grounding_dino": dino_result, "grounding_dino_sam21": sam_result}
            writer.write(json.dumps(row) + "\n"); existing[item["identity"]] = row
            if number % 100 == 0: print(f"processed {number}/{len(inputs)}", flush=True)
    rows = [existing[item["identity"]] for item in inputs]
    methods = ["grounding_dino"] + ([] if sam is None else ["grounding_dino_sam21"])
    summary = {method: summarize(rows, method) for method in methods}
    if sam is not None:
        delta = [row["grounding_dino_sam21"]["metrics"]["iou"] - row["grounding_dino"]["metrics"]["iou"] for row in rows]
        summary["sam21_minus_dino"] = {"iou_delta": distribution(delta),
                                       "improved_fraction": float(np.mean(np.asarray(delta) > 0)),
                                       "degraded_fraction": float(np.mean(np.asarray(delta) < 0))}
    manifest = {
        "status": "complete", "protocol": "aircraft_oss_bbox_audit_v1",
        "images": len(rows), "split": str(args.variants), "caption": dino.caption,
        "selection": "highest score among all 900 Grounding DINO queries; no confidence threshold or official-box tuning",
        "sam": "SAM2.1 Hiera-L; DINO box prompt; multimask_output=False; thresholded mask tight XYXY box",
        "official_boxes": "inclusive one-indexed Aircraft boxes converted to continuous zero-indexed [x1-1,y1-1,x2,y2]",
        "grounding_revision": (args.grounding_root / "UPSTREAM_REVISION").read_text().strip(),
        "sam_revision": (args.sam_root / "UPSTREAM_REVISION").read_text().strip(),
        "grounding_checkpoint_sha256": sha256(args.grounding_checkpoint),
        "text_encoder_root": None if args.text_encoder_root is None else str(args.text_encoder_root.resolve()),
        "sam_checkpoint_sha256": None if sam is None else sha256(args.sam_checkpoint),
        "grounding_load_audit": dino.load_audit, "summary": summary,
        "predictions": str(output_jsonl.resolve()),
    }
    (args.output_root / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
