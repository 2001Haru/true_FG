"""Write the immutable generation contract for a fine-grained CoDA arm."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--classes", required=True, type=int)
    parser.add_argument("--ipc", required=True, type=int)
    parser.add_argument("--generation-seed", required=True, type=int)
    parser.add_argument("--feature-space", required=True, choices=("vae", "dinov2"))
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--dino-model-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    model_index = args.model_root / "sdxl-base" / "model_index.json"
    if not model_index.is_file():
        raise FileNotFoundError(model_index)
    dino_weights = args.dino_model_root / "model.safetensors"
    dino_config = args.dino_model_root / "config.json"
    dino_preprocessor = args.dino_model_root / "preprocessor_config.json"
    if args.feature_space == "dinov2" and not dino_weights.is_file():
        raise FileNotFoundError(dino_weights)
    if args.feature_space == "dinov2" and not dino_preprocessor.is_file():
        raise FileNotFoundError(dino_preprocessor)
    revision = subprocess.check_output(
        ["git", "-C", str(args.repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    source_files = [
        args.repo_root / "CoDA" / name
        for name in (
            "CoDA_main.py",
            "Loadmodel.py",
            "CoDA_SDXLBasePipeline.py",
            "get_features.py",
            "postprocess.py",
            "generated.py",
            "fg_prompts.py",
        )
    ]
    payload = {
        "status": "frozen",
        "method": "CoDA fine-grained adapter",
        "paper": "arXiv:2512.03844v1",
        "official_code": "https://github.com/zzzlt422/CoDA",
        "official_code_revision": "0c12e44c185929687a27825de4e1efca20700231",
        "git_revision": revision,
        "dataset": args.dataset_name,
        "spec": args.spec,
        "classes": args.classes,
        "ipc": args.ipc,
        "feature_space": args.feature_space,
        "data_dir": str(args.data_dir.resolve()),
        "shared_feature_cache_root": str(args.cache_root.resolve()),
        "generation_seed": args.generation_seed,
        "clustering_feature_encoder": (
            "SDXL fp16-fix VAE flattened latent (65536D), input 1024x1024"
            if args.feature_space == "vae" else
            "DINOv2-base final normalized CLS token (768D), Resize256+CenterCrop224"
        ),
        "sdxl_guidance_feature": (
            "same VAE latent used for clustering"
            if args.feature_space == "vae" else
            "path-aligned SDXL VAE latent of the source image selected in DINOv2 space"
        ),
        "umap": {
            "dimensions": "min(50, class_sample_count - 2)",
            "n_neighbors": 5,
            "min_dist": 0.0,
            "random_state": 42,
        },
        "hdbscan": {"min_cluster_size": 2, "min_samples": 1},
        "postprocess": "paper Algorithm 2; FG outlier threshold requires count >= missing IPC",
        "per_source_image_cluster_provenance": (
            "initial HDBSCAN label/noise/probability/outlier score, final disposition, "
            "representative selection origin and slot"
        ),
        "per_generated_image_provenance": (
            "generated SHA-256 and seed/GPU joined to the exact representative source image"
        ),
        "generator": "SDXL Base 1.0",
        "model_root": str(args.model_root.resolve()),
        "model_index_sha256": sha256(model_index),
        "dino_model_root": (
            str(args.dino_model_root.resolve()) if args.feature_space == "dinov2" else None
        ),
        "dino_model_sha256": (
            sha256(dino_weights) if args.feature_space == "dinov2" else None
        ),
        "dino_config_sha256": (
            sha256(dino_config) if args.feature_space == "dinov2" else None
        ),
        "dino_preprocessor_sha256": (
            sha256(dino_preprocessor) if args.feature_space == "dinov2" else None
        ),
        "sampler": "DPM++ Karras",
        "inference_steps": 25,
        "denoising_factor": 1.0,
        "guide_t_percent": 0.9,
        "prior_injection_steps_nominal_formula": 2.5,
        "prior_injection_steps_discrete": 3,
        "coda_guided_steps_discrete": 22,
        "cfg_scale": 5.0,
        "coda_guidance_scale_gamma": 0.05,
        "negative_prompt": None,
        "prompt": "canonical fine-grained class name only",
        "per_image_seed_formula": (
            "generation_seed + visible_gpu_id*10000 + class_id*IPC + 1000 + image_id"
        ),
        "generation_gpu_count": 2,
        "internal_generation_size": [1024, 1024],
        "saved_image_size": [224, 224],
        "source_sha256": {str(path.relative_to(args.repo_root)): sha256(path) for path in source_files},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
