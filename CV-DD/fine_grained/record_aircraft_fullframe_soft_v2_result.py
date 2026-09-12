"""Audit Aircraft IPC3 full-frame resize+flip, no-CutMix Soft-v2 result."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from audit_result import audit_payload


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def sha256(path):
    d=hashlib.sha256()
    with Path(path).open("rb") as h:
        for c in iter(lambda:h.read(1024*1024),b""): d.update(c)
    return d.hexdigest()
def expect(a,b,label):
    if a!=b: raise RuntimeError(f"{label}: {a!r} != {b!r}")
def close(a,b,label):
    if not math.isclose(float(a),b,rel_tol=0,abs_tol=1e-12): raise RuntimeError(f"{label}: {a!r} != {b!r}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result",required=True,type=Path);p.add_argument("--source-image-result",required=True,type=Path)
    p.add_argument("--fkd",required=True,type=Path);p.add_argument("--teacher",required=True,type=Path)
    p.add_argument("--method",required=True,choices=("random_real","original_rded"));p.add_argument("--source-seed",required=True,type=int)
    p.add_argument("--student-seed",required=True,type=int);a=p.parse_args()
    r,s=load(a.result),load(a.source_image_result);audit_payload(r,100,3333)
    expect(r["student_protocol_name"],"standard_protocol_v2","protocol");expect(r["student_initialization"],"imagenet-v1","initialization")
    expect(r["student_seed"],a.student_seed,"seed");expect(r["synthetic_data_path"],s["synthetic_data_path"],"images")
    expect(r["fkd_path"],str(a.fkd.resolve()),"FKD");expect(r["mix_type"],None,"mix");expect(r["batch_size"],20,"batch")
    expect(r["gradient_accumulation_steps"],2,"accumulation");expect(r["epochs"],400,"epochs")
    close(r["backbone_learning_rate"],1e-4,"backbone LR");close(r["head_learning_rate"],1e-3,"head LR");close(r["weight_decay"],1e-5,"WD")
    expect(r["scheduler"],"cosine_annealing","scheduler");expect(r["scheduler_t_max"],400,"T_max");close(r["temperature"],20,"temperature")
    m=load(a.fkd/"relabel_manifest.json");expect(m["status"],"complete","Relabel");expect(m["teacher_mode"],"train","BSSL")
    expect(m["full_image_resize"],True,"full image");expect(m["mix_type"],None,"Relabel mix");expect(m["train_view"],"Resize224(full frame) + HorizontalFlip","view")
    expect(m["teacher_sha256"],sha256(a.teacher),"Teacher hash")
    r["fullframe_no_cutmix_soft_v2"]={"status":"complete","method":a.method,"source_seed":a.source_seed,
        "student_seed":a.student_seed,"source_image_result":str(a.source_image_result.resolve()),
        "source_image_result_sha256":sha256(a.source_image_result),"teacher_sha256":sha256(a.teacher)}
    t=a.result.with_suffix(".json.tmp");t.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n",encoding="utf-8");os.replace(t,a.result)


if __name__=="__main__":main()
