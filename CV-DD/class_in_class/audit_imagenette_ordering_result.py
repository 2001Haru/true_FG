"""Strictly audit one ordering-hierarchy training result."""

import argparse
import json
import math
from pathlib import Path

from imagenette_entropy_protocol import file_sha256, lr_for_epoch
from train_imagenette_ordering_hierarchy import EVAL_EPOCHS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--arm", choices=("O", "B", "A", "Aprime"), required=True)
    parser.add_argument("--selection-seed", choices=tuple(range(3,12)), required=True, type=int)
    parser.add_argument("--student-seed", choices=(42,43,44), required=True, type=int)
    args = parser.parse_args()
    p = json.loads(args.result.read_text(encoding="utf-8"))
    expected = {
        "status":"complete","protocol":"imagenette_ordering_hierarchy_v1","arm":args.arm,
        "dataset":"imagenet-nette","classes":10,"ipc":10,"train_images":100,"test_images":3925,
        "selection_seed":args.selection_seed,"student_seed":args.student_seed,
        "student_model":"CoDA_released_ResNetAP10","student_initialization":"random",
        "image_size":256,"optimizer":"SGD","initial_lr":.01,"momentum":.9,"weight_decay":.0005,
        "batch_size":64,"actual_batch_sizes_per_epoch":[64,36],"gradient_accumulation_steps":1,
        "drop_last":False,"epochs":2000,"updates_completed":4000,"scheduler":"released_multistep_epoch",
        "lr_milestones_after_epochs":[1333,1666],"lr_gamma":.2,"cutmix":False,"mixup":False,
        "evaluation_epochs":list(EVAL_EPOCHS),"evaluation_count":12,
        "teacher_temperature":1.0,"student_temperature":1.0,"teacher_mode":"eval_frozen_no_grad",
        "teacher_and_student_view_pixels_identical":True,"high_entropy_images_per_epoch":50,
        "low_entropy_hard_images_per_epoch":50,"primary_metric":"final_true_label_nll_at_update_4000",
    }
    errors=[]
    for key,value in expected.items():
        actual=p.get(key)
        if isinstance(value,float):
            if actual is None or not math.isclose(float(actual),value,abs_tol=1e-12): errors.append(f"{key} mismatch")
        elif actual!=value: errors.append(f"{key} mismatch")
    history=p.get("test_history",[])
    if [row.get("epoch") for row in history]!=list(EVAL_EPOCHS): errors.append("evaluation cadence mismatch")
    for row in history:
        if row.get("update")!=2*row["epoch"] or not math.isclose(row.get("lr",-1),lr_for_epoch(row["epoch"]),abs_tol=1e-12): errors.append("epoch/update/lr mismatch")
    if not history or p.get("final_loss")!=history[-1].get("test_loss") or p.get("final_top1")!=history[-1].get("test_top1"): errors.append("Final mismatch")
    invariant=p.get("label_invariant_audit",{})
    if invariant.get("views")!=100000 or invariant.get("teacher_correctness_mismatches")!=0: errors.append("invariant coverage/correctness mismatch")
    for key,value in invariant.get("maximum_absolute_differences",{}).items():
        if float(value)>1e-5: errors.append(f"invariant failed: {key}")
    if p.get("teacher_online_original_statistics",{}).get("views")!=100000: errors.append("Teacher statistics coverage mismatch")
    if len(p.get("per_class_final",[]))!=10: errors.append("per-class result missing")
    for key,hash_key in (("protocol_spec","protocol_spec_sha256"),("student_model_source","student_model_source_sha256"),("train_manifest","train_manifest_sha256"),("ordering_templates","ordering_templates_sha256"),("teacher_checkpoint","teacher_checkpoint_sha256"),("final_checkpoint","final_checkpoint_sha256")):
        path=Path(p.get(key,""))
        if not path.is_file() or file_sha256(path)!=p.get(hash_key): errors.append(f"provenance mismatch: {key}")
    if errors: raise RuntimeError("ordering result audit failed:\n"+"\n".join(errors))
    print(json.dumps({"status":"complete","result":str(args.result.resolve())}))


if __name__ == "__main__": main()

