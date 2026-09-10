"""Audit and annotate a RandomReal + fixed-Teacher FKD + Student-v2 result."""

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

from audit_result import audit_payload


def load(path):
    if not path.is_file(): raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def expect(actual,expected,label):
    if actual!=expected: raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def expect_float(actual,expected,label,tol=1e-12):
    if actual is None or not math.isclose(float(actual),expected,rel_tol=0,abs_tol=tol):
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result",required=True,type=Path)
    parser.add_argument("--selection-manifest",required=True,type=Path)
    parser.add_argument("--teacher-dir",required=True,type=Path)
    parser.add_argument("--fkd-dir",required=True,type=Path)
    parser.add_argument("--ipc",required=True,type=int,choices=(1,3,5))
    parser.add_argument("--selection-seed",required=True,type=int,choices=(0,1,2))
    parser.add_argument("--student-seed",required=True,type=int,choices=(42,43,44))
    args=parser.parse_args(); result=load(args.result); manifest=load(args.selection_manifest)
    audit_payload(result,100,3333)
    expect(manifest["status"],"complete","selection status")
    expect(manifest["dataset"],"A_imsize224","selection dataset")
    expect(manifest["ipc"],args.ipc,"selection IPC")
    expect(manifest["selection_seed"],args.selection_seed,"selection seed")
    expect(manifest["selected_images"],100*args.ipc,"selected images")
    expect(result["synthetic_data_path"],str(Path(manifest["selected_root"]).resolve()),"selected path")
    expect(result["fkd_path"],str(args.fkd_dir.resolve()),"FKD path")
    expect(result["student_protocol_name"],"standard_protocol_v2","protocol")
    expect(result["student_initialization"],"imagenet-v1","initialization")
    expect(result["student_seed"],args.student_seed,"Student seed")
    expect(result["training_target"],"fkd_soft_label","training target")
    expect(result["optimizer"],"adamw","optimizer")
    expect(result["optimizer_betas"],[0.9,0.999],"betas")
    expect_float(result["optimizer_eps"],1e-8,"eps")
    expect_float(result["backbone_learning_rate"],1e-4,"backbone LR")
    expect_float(result["head_learning_rate"],1e-3,"head LR")
    expect_float(result["weight_decay"],1e-5,"matrix WD")
    expect(result["scheduler"],"cosine_annealing","scheduler")
    expect(result["scheduler_t_max"],400,"T_max")
    expect_float(result["scheduler_eta_min"],0,"eta_min")
    if any(abs(float(x))>1e-12 for x in result["final_scheduler_lrs"]): raise RuntimeError("final LR nonzero")
    expect(result["epochs"],400,"epochs"); expect(result["batch_size"],20,"batch")
    expect(result["gradient_accumulation_steps"],2,"accumulation")
    expect_float(result["temperature"],20,"temperature")
    expect(result["temperature_squared_multiplier"],False,"T squared")
    expect(result["hard_cross_entropy_mixture"],False,"hard CE")
    expect(result["mix_type"],"cutmix","CutMix")
    expect(result["dataloader_workers"],8,"workers"); expect(result["persistent_workers"],True,"persistent")
    groups={g["group_name"]:g for g in result["optimizer_parameter_groups"]}
    expect(set(groups),{"backbone_decay","backbone_no_decay","head_decay","head_no_decay"},"groups")

    teacher=load(args.teacher_dir/"complete.json"); teacher_ckpt=args.teacher_dir/"ResNet18.pth"
    expect(teacher["status"],"complete","Teacher status"); expect(teacher["seed"],42,"Teacher seed")
    relabel=load(args.fkd_dir/"relabel_manifest.json"); fkd=load(args.fkd_dir/"fkd_audit.json")
    expect(relabel["status"],"complete","Relabel status"); expect(relabel["ipc"],args.ipc,"Relabel IPC")
    expect(relabel["teacher_mode"],"train","Teacher mode"); expect(relabel["epochs"],400,"Relabel epochs")
    expect(relabel["batch_size"],20,"Relabel batch"); expect(relabel["workers"],8,"Relabel workers")
    expect(relabel["persistent_workers"],False,"Relabel persistent")
    expect_float(relabel["temperature"],20,"Relabel temperature"); expect(relabel["mix_type"],"cutmix","Relabel CutMix")
    expect(relabel["teacher_sha256"],sha256(teacher_ckpt),"Teacher hash")
    expect(fkd["status"],"complete","FKD status"); expect(fkd["images"],100*args.ipc,"FKD images")

    protocol_path=Path(__file__).with_name("standard_v2_protocol.json"); repo=Path(__file__).resolve().parents[2]
    result["standard_protocol"]={"name":"fine_grained_sre2l_standard_protocol","version":"v2",
        "git_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),
        "dataset":"A_imsize224","teacher_seed":42,"selection_seed":args.selection_seed,
        "ipc":args.ipc,"student_seed":args.student_seed,"protocol_definition":str(protocol_path.resolve()),
        "protocol_definition_sha256":sha256(protocol_path)}
    result["random_real_soft_v2"]={"selection_algorithm":manifest["selection_algorithm"],
        "selection_manifest":str(args.selection_manifest.resolve()),"selection_manifest_sha256":sha256(args.selection_manifest),
        "selected_tree_sha256":manifest["selected_tree_sha256"],"teacher_checkpoint":str(teacher_ckpt.resolve()),
        "teacher_checkpoint_sha256":sha256(teacher_ckpt),"teacher_seed":42,
        "relabel_manifest":str((args.fkd_dir/"relabel_manifest.json").resolve()),
        "relabel_manifest_sha256":sha256(args.fkd_dir/"relabel_manifest.json"),
        "fkd_audit":str((args.fkd_dir/"fkd_audit.json").resolve()),"fkd_audit_sha256":sha256(args.fkd_dir/"fkd_audit.json")}
    temporary=args.result.with_suffix(".json.tmp"); temporary.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"); os.replace(temporary,args.result)
    print(json.dumps({"status":"complete","result":str(args.result.resolve())}))


if __name__=="__main__": main()
