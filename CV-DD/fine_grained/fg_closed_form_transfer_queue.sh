#!/usr/bin/env bash
set -euo pipefail
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ROOT=/linxi/true_FG
FG="$ROOT/CV-DD/fine_grained"
EXP=/linxi/dataset/FGDD_closed_form_transfer/v1
STAGE="$EXP/stage"
RANDOM=/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard
DATA=/linxi/dataset/FG_SRe2L_repro/v1/datasets
TEACHERS=/linxi/dataset/FG_SRe2L_standard/v1/teachers
WEIGHTS=/linxi/models/torchvision/resnet18-f37072fd.pth
OLD=/linxi/dataset/FGDD_transfer/cub_cars_k3a1_v1
OLD_STAGE=/tmp/fgdd_cub_cars_k3a1_v1
mkdir -p "$EXP"/{logs,audits,results,summary,status,locks} "$STAGE"/{bbox,construction,fit,fkd,post_eval}
exec 9>"$EXP/locks/queue.lock"; flock -n 9 || exit 75
trap 's=$?; rm -f "$EXP/status/running"; if ((s)); then date --iso-8601=seconds >"$EXP/status/failed"; fi' EXIT
rm -f "$EXP/status/failed" "$EXP/status/complete"; date --iso-8601=seconds >"$EXP/status/running"

# Freeze all three official-box maps in the persistent experiment root.
python "$FG/prepare_aircraft_bbox_manifest.py" \
  --prepared-train "$DATA/A_imsize224/train" \
  --raw-images /linxi/dataset/FD2/raw/fgvc-aircraft-2013b/data/images \
  --boxes /linxi/dataset/FD2/raw/fgvc-aircraft-2013b/data/images_box.txt \
  --output "$STAGE/bbox/A_imsize224_bbox_manifest.json" >"$EXP/logs/bbox_A.log" 2>&1
cp "$OLD_STAGE/bbox/CUB_imsize224_bbox_manifest.json" "$STAGE/bbox/"
cp "$OLD_STAGE/bbox/SC_imsize224_bbox_manifest.json" "$STAGE/bbox/"

classes(){ case "$1" in A_imsize224) echo 100;; CUB_imsize224) echo 200;; SC_imsize224) echo 196;; esac; }
batch(){ [[ $1 == SC_imsize224 ]] && echo 14 || echo 20; }
mean(){ case "$1" in A_imsize224) echo '.4865 .5177 .5425';; CUB_imsize224) echo '.4857 .4994 .4326';; SC_imsize224) echo '.4708 .4601 .4551';; esac; }
std(){ case "$1" in A_imsize224) echo '.2124 .2051 .2375';; CUB_imsize224) echo '.2260 .2215 .2595';; SC_imsize224) echo '.2885 .2879 .2962';; esac; }

# name:dataset:storage_ipc:k:axis.  Axis and k are the frozen closed-form rule;
# Cars k3 is retained only as the requested axis-isolation diagnostic.
CONFIGS=(
  A_ipc1_k3:A_imsize224:1:3:row
  CUB_ipc1_k2:CUB_imsize224:1:2:column
  SC_ipc1_k2:SC_imsize224:1:2:row
  CUB_ipc3_k2:CUB_imsize224:3:2:column
  SC_ipc3_k2:SC_imsize224:3:2:row
  SC_ipc3_k3:SC_imsize224:3:3:row
)

for spec in "${CONFIGS[@]}"; do
  IFS=: read -r name dataset ipc k axis <<<"$spec"; out="$STAGE/construction/$name"; mkdir -p "$out"
  read -ra mu <<<"$(mean "$dataset")"; read -ra sigma <<<"$(std "$dataset")"
  python "$FG/prepare_fg_rule_pack.py" \
    --selection-manifest "$RANDOM/manifests/$dataset/rseed0/ipc10.json" \
    --bbox-manifest "$STAGE/bbox/${dataset}_bbox_manifest.json" --axis "$axis" --k "$k" \
    --storage-ipc "$ipc" --phi .106 --output-root "$out" --mean "${mu[@]}" --std "${sigma[@]}" \
    >"$EXP/logs/pack_${name}.log" 2>&1
done

for dataset in A_imsize224 CUB_imsize224 SC_imsize224; do
  python "$FG/prepare_fg_public600_manifest.py" --bbox-manifest "$STAGE/bbox/${dataset}_bbox_manifest.json" \
    --selection-manifest "$RANDOM/manifests/$dataset/rseed0/ipc10.json" \
    --output "$STAGE/fit/${dataset}_public600.json" >"$EXP/logs/public600_${dataset}.log" 2>&1
done

fit(){
  local name=$1 dataset=$2 gpu=$3
  CUDA_VISIBLE_DEVICES=$gpu python "$FG/fit_fg_fir_preemphasis.py" \
    --public-manifest "$STAGE/fit/${dataset}_public600.json" \
    --bbox-manifest "$STAGE/bbox/${dataset}_bbox_manifest.json" \
    --base-manifest "$STAGE/construction/$name/base_manifest.json" --imagenet-weights "$WEIGHTS" \
    --output-manifest "$STAGE/construction/$name/fir_manifest.json" \
    --audit "$EXP/audits/fir_${name}.json" >"$EXP/logs/fit_${name}.log" 2>&1
}
fit A_ipc1_k3 A_imsize224 0 & p0=$!; fit CUB_ipc3_k2 CUB_imsize224 1 & p1=$!; wait $p0; wait $p1
fit SC_ipc3_k2 SC_imsize224 0 & p0=$!; fit SC_ipc3_k3 SC_imsize224 1 & p1=$!; wait $p0; wait $p1

# Reuse the dataset/axis/k fit at IPC1; the public600 fit does not depend on
# storage IPC or selected source identities.
python - "$STAGE" <<'PY'
import copy,json,sys
from pathlib import Path
root=Path(sys.argv[1])
for source,target in (('CUB_ipc3_k2','CUB_ipc1_k2'),('SC_ipc3_k2','SC_ipc1_k2')):
    fitted=json.load(open(root/'construction'/source/'fir_manifest.json'))
    base=json.load(open(root/'construction'/target/'base_manifest.json'))
    base['fir_preemphasis']=copy.deepcopy(fitted['fir_preemphasis'])
    base['protocol']+='+fir_preemphasis_v1'
    (root/'construction'/target/'fir_manifest.json').write_text(json.dumps(base,indent=2)+'\n')
PY

# Static gates: exact closed-form axes/k, positive density, complete packing,
# no source duplication, and exact metadata compatibility with the old Cars U9.
python - "$STAGE" "$OLD_STAGE" "$EXP/audits/preflight.json" <<'PY'
import json,sys
from pathlib import Path
stage,old,out=map(Path,sys.argv[1:]); expected={
'A_ipc1_k3':('A_imsize224',1,3,'row'),'CUB_ipc1_k2':('CUB_imsize224',1,2,'column'),
'SC_ipc1_k2':('SC_imsize224',1,2,'row'),'CUB_ipc3_k2':('CUB_imsize224',3,2,'column'),
'SC_ipc3_k2':('SC_imsize224',3,2,'row'),'SC_ipc3_k3':('SC_imsize224',3,3,'row')}
rows=[]
for name,(dataset,ipc,k,axis) in expected.items():
    base=json.load(open(stage/'construction'/name/'base_manifest.json'))
    fir=json.load(open(stage/'construction'/name/'fir_manifest.json'))
    assert base['dataset']==dataset and base['storage_ipc']==ipc and base['sources_per_parent']==k and base['compression_axis']==axis
    assert base['parents']==base['class_count']*ipc and base['unique_sources']==base['class_count']*ipc*k
    paths=[]
    for parent in base['records']:
        assert len(parent['sources'])==k and sum(x['encoded_height'] for x in parent['sources'])==224
        for source in parent['sources']:
            assert min(source['density'])>0 and abs(sum(source['density'])-source['encoded_height'])<1e-7
            paths.append(source['reference_path'])
    assert len(paths)==len(set(paths))
    assert fir['fir_preemphasis']['axis']==axis
    rows.append({'name':name,'axis':axis,'k':k,'storage_ipc':ipc,'background_minimum':base['background_minimum'],
                 'mean_axis_fraction':base['mean_selected_axis_box_fraction'],'subject_density':base['subject_density_stats']})
new=json.load(open(stage/'construction'/'SC_ipc3_k3'/'base_manifest.json'))
prior=json.load(open(old/'construction'/'SC_imsize224'/'base_manifest.json'))
new_paths=[[x['reference_path'] for x in row['sources']] for row in new['records']]
old_paths=[[x['reference_path'] for x in row['sources']] for row in prior['records']]
assert new_paths==old_paths
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'status':'complete','configs':rows,'cars_k3_old_u9_metadata_compatible':True},indent=2)+'\n')
PY

relabel(){
  local name=$1 dataset=$2 arm=$3 source=$4 gpu=$5 manifest mode output bs c
  bs=$(batch "$dataset"); c=$(classes "$dataset"); output="$STAGE/fkd/$name/$arm"
  manifest="$STAGE/construction/$name/base_manifest.json"; mode=reference
  [[ $arm == f1 ]] && { manifest="$STAGE/construction/$name/fir_manifest.json"; mode=compressed_fir; }
  [[ $arm == r0 ]] && manifest="$STAGE/construction/$name/anchor_manifest.json"
  [[ $source == self ]] && source="$output"
  CUDA_VISIBLE_DEVICES=$gpu python "$FG/relabel_aircraft_six_source_fir.py" --manifest "$manifest" \
    --source-fkd "$source" --output-fkd "$output" --teacher "$TEACHERS/$dataset/tseed42/ResNet18.pth" \
    --image-mode "$mode" --batch-size "$bs" --classes "$c" --workers 6 >"$EXP/logs/relabel_${name}_${arm}.log" 2>&1
}

# IPC3 expansions.
relabel CUB_ipc3_k2 CUB_imsize224 u6 self 0 & p0=$!
relabel SC_ipc3_k2 SC_imsize224 u6 self 1 & p1=$!; wait $p0; wait $p1
relabel CUB_ipc3_k2 CUB_imsize224 f1 "$STAGE/fkd/CUB_ipc3_k2/u6" 0 & p0=$!
relabel SC_ipc3_k2 SC_imsize224 f1 "$STAGE/fkd/SC_ipc3_k2/u6" 1 & p1=$!; wait $p0; wait $p1
relabel SC_ipc3_k3 SC_imsize224 f1 "$OLD_STAGE/fkd/SC_imsize224/u9" 0

# IPC1 triads.  Uk establishes the trajectory; F1 and the duplicated-rank0 R0
# control replay the exact source-index/flip/CutMix metadata.
for spec in A_ipc1_k3:A_imsize224 CUB_ipc1_k2:CUB_imsize224 SC_ipc1_k2:SC_imsize224; do
  IFS=: read -r name dataset <<<"$spec"
  relabel "$name" "$dataset" uk self 0
  relabel "$name" "$dataset" f1 "$STAGE/fkd/$name/uk" 0 & p0=$!
  relabel "$name" "$dataset" r0 "$STAGE/fkd/$name/uk" 1 & p1=$!; wait $p0; wait $p1
done

train(){
  local name=$1 dataset=$2 ipc=$3 arm=$4 seed=$5 gpu=$6 manifest mode fkd result output bs
  bs=$(batch "$dataset"); manifest="$STAGE/construction/$name/base_manifest.json"; mode=reference
  [[ $arm == f1 ]] && { manifest="$STAGE/construction/$name/fir_manifest.json"; mode=compressed_fir; }
  [[ $arm == r0 ]] && manifest="$STAGE/construction/$name/anchor_manifest.json"
  fkd="$STAGE/fkd/$name/$arm"; result="$EXP/results/${name}_${arm}_sseed${seed}.json"
  output="$STAGE/post_eval/${name}_${arm}_sseed${seed}"; [[ -f $result ]] && return; mkdir -p "$output"
  CUDA_VISIBLE_DEVICES=$gpu python "$ROOT/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc "$ipc" \
    --exp-name "closed_${name}_${arm}_s${seed}" --original-data-path "$STAGE/construction/$name/packed" \
    --fkd-path "$fkd" --paired-source-manifest "$manifest" --paired-source-mode "$mode" --output-dir "$output" \
    --batch-size "$bs" --epochs 400 --dataset-name "$dataset" --gradient-accumulation-steps 2 --mix-type cutmix \
    --workers 6 --persistent-workers --fkd_seed 42 --train-seed "$seed" --temperature 20 \
    --student-initialization imagenet-v1 --student-protocol-name fg_closed_form_soft_pff \
    --adamw-weight-decay 1e-5 --adamw-beta1 .9 --adamw-beta2 .999 --adamw-eps 1e-8 \
    --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 --cosine-t-max 400 --cosine-eta-min 0 \
    --val-dir "$DATA/$dataset/test" --disable-wandb --per-class-output "$result" >"$EXP/logs/${name}_${arm}_sseed${seed}.log" 2>&1
}

JOBS=()
for arm in f1; do for seed in 42 43 44; do JOBS+=("SC_ipc3_k3:SC_imsize224:3:$arm:$seed"); done; done
for spec in CUB_ipc3_k2:CUB_imsize224 SC_ipc3_k2:SC_imsize224; do IFS=: read -r name dataset <<<"$spec"; for arm in u6 f1; do for seed in 42 43 44; do JOBS+=("$name:$dataset:3:$arm:$seed"); done; done; done
for spec in A_ipc1_k3:A_imsize224 CUB_ipc1_k2:CUB_imsize224 SC_ipc1_k2:SC_imsize224; do IFS=: read -r name dataset <<<"$spec"; for arm in r0 uk f1; do for seed in 42 43 44; do JOBS+=("$name:$dataset:1:$arm:$seed"); done; done; done
for ((i=0;i<${#JOBS[@]};i+=4)); do
  pids=(); for ((j=0;j<4 && i+j<${#JOBS[@]};j++)); do
    IFS=: read -r name dataset ipc arm seed <<<"${JOBS[i+j]}"; train "$name" "$dataset" "$ipc" "$arm" "$seed" $((j%2)) & pids+=("$!")
  done
  failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; ((failed==0))
done

python - "$EXP" "$OLD" <<'PY'
import json,statistics,sys
from pathlib import Path
root,old=map(Path,sys.argv[1:]); groups={}; comparisons={}
def stats(values):
    values=list(map(float,values));return {'mean':statistics.mean(values),'sample_sd':statistics.stdev(values),'values':values,'positive_count':sum(x>0 for x in values)}
def load(name,arm):return {s:json.load(open(root/'results'/f'{name}_{arm}_sseed{s}.json')) for s in (42,43,44)}
def record(name,arms):
    rows={arm:load(name,arm) for arm in arms}
    for arm,values in rows.items():groups[f'{name}_{arm}']={'best':stats(values[s]['best_top1'] for s in values),'final':stats(values[s]['final_epoch_top1'] for s in values),'best_epochs':[values[s]['best_epoch'] for s in values]}
    return rows
for name,arms in [('SC_ipc3_k3',('f1',)),('CUB_ipc3_k2',('u6','f1')),('SC_ipc3_k2',('u6','f1')),
                  ('A_ipc1_k3',('r0','uk','f1')),('CUB_ipc1_k2',('r0','uk','f1')),('SC_ipc1_k2',('r0','uk','f1'))]:
    rows=record(name,arms)
    if {'r0','uk','f1'}<=rows.keys():
        comparisons[name]={key:stats(rows[a][s]['final_epoch_top1']-rows[b][s]['final_epoch_top1'] for s in (42,43,44))
          for key,a,b in [('instance_uk_minus_r0','uk','r0'),('encoding_uk_minus_f1','uk','f1'),('net_f1_minus_r0','f1','r0')]}
    elif {'u6','f1'}<=rows.keys():comparisons[name]={'encoding_u6_minus_f1':stats(rows['u6'][s]['final_epoch_top1']-rows['f1'][s]['final_epoch_top1'] for s in (42,43,44))}
prior=json.load(open(old/'summary'/'summary.json'))
new=load('SC_ipc3_k3','f1'); prior_f1=prior['groups']['SC_imsize224_f1']['final']['values']
comparisons['SC_ipc3_k3']['row_minus_old_column_f1']=stats(new[s]['final_epoch_top1']-prior_f1[i] for i,s in enumerate((42,43,44)))
out={'status':'complete','protocol':'fg_closed_form_transfer_v1','phi':.106,'tau':.6,'groups':groups,'comparisons':comparisons,
     'expected_fkd':14,'expected_students':42,'preflight':json.load(open(root/'audits'/'preflight.json'))}
(root/'summary'/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
PY

rm -f "$EXP/status/running"; date --iso-8601=seconds >"$EXP/status/complete"
