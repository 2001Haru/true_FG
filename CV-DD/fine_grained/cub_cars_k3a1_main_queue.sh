#!/usr/bin/env bash
set -euo pipefail
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ROOT=/linxi/true_FG;FG="$ROOT/CV-DD/fine_grained"
EXP=/linxi/dataset/FGDD_transfer/cub_cars_k3a1_v1;STAGE=/tmp/fgdd_cub_cars_k3a1_v1
RANDOM_ROOT=/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard
DATA=/linxi/dataset/FG_SRe2L_repro/v1/datasets;TEACHERS=/linxi/dataset/FG_SRe2L_standard/v1/teachers
WEIGHTS=/linxi/models/torchvision/resnet18-f37072fd.pth
mkdir -p "$EXP"/{logs,audits,results,summary,status,locks} "$STAGE"/{construction,fkd,post_eval,fit}
exec 9>"$EXP/locks/main_phase.lock";flock -n 9||exit 75
trap 's=$?;rm -f "$EXP/status/main.running";if((s));then date --iso-8601=seconds>"$EXP/status/main.failed";fi' EXIT
rm -f "$EXP/status/main.failed" "$EXP/status/main.complete";date --iso-8601=seconds>"$EXP/status/main.running"

while [[ ! -f "$EXP/status/r0.complete" ]];do [[ -f "$EXP/status/r0.failed" ]]&&exit 1;sleep 30;done
cp "$STAGE/bbox/summary.json" "$EXP/audits/bbox_summary.json";cp "$STAGE/bbox/axis_feasibility.json" "$EXP/audits/axis_feasibility.json"

checkpoint(){ find "$STAGE/post_eval/${1}_r0_sseed${2}" -name checkpoint.pth.tar -type f|head -1; }
CUDA_VISIBLE_DEVICES=0 python "$FG/audit_fg_axis_p5.py" --bbox-manifest "$STAGE/bbox/CUB_imsize224_bbox_manifest.json" \
 --selection-manifest "$RANDOM_ROOT/manifests/CUB_imsize224/rseed0/ipc10.json" --classes 200 --mean .4857 .4994 .4326 --std .2260 .2215 .2595 \
 --checkpoint "$(checkpoint CUB_imsize224 42)" --checkpoint "$(checkpoint CUB_imsize224 43)" --checkpoint "$(checkpoint CUB_imsize224 44)" \
 --output "$EXP/audits/p5_CUB_imsize224.json" --device cuda:0 >"$EXP/logs/p5_CUB.log" 2>&1&p0=$!
CUDA_VISIBLE_DEVICES=1 python "$FG/audit_fg_axis_p5.py" --bbox-manifest "$STAGE/bbox/SC_imsize224_bbox_manifest.json" \
 --selection-manifest "$RANDOM_ROOT/manifests/SC_imsize224/rseed0/ipc10.json" --classes 196 --mean .4708 .4601 .4551 --std .2885 .2879 .2962 \
 --checkpoint "$(checkpoint SC_imsize224 42)" --checkpoint "$(checkpoint SC_imsize224 43)" --checkpoint "$(checkpoint SC_imsize224 44)" \
 --output "$EXP/audits/p5_SC_imsize224.json" --device cuda:0 >"$EXP/logs/p5_Cars.log" 2>&1&p1=$!;wait $p0;wait $p1

python - "$EXP/audits/axis_feasibility.json" "$EXP/audits/p5_CUB_imsize224.json" "$EXP/audits/p5_SC_imsize224.json" "$EXP/audits/axis_decision.json" <<'PY'
import json,sys
from pathlib import Path
feas=json.load(open(sys.argv[1]));out={'status':'complete','adaptive_rule':'requires >=0.10 feasible-source improvement over best fixed axis and tall/wide P5 sign flip','datasets':{}}
for name,p5path in (('CUB_imsize224',sys.argv[2]),('SC_imsize224',sys.argv[3])):
 p5=json.load(open(p5path));f=feas['datasets'][name]['at_transferred_background'];best_fixed=max(f['row']['feasible_source_fraction'],f['column']['feasible_source_fraction'])
 gain=f['adaptive_same_axis_slot']['feasible_source_fraction']-best_fixed;tall=p5['groups']['tall']['column_minus_row']['mean'];wide=p5['groups']['wide']['column_minus_row']['mean'];flip=tall*wide<0
 adaptive=gain>=.10 and flip;overall=p5['groups']['all']['column_minus_row']['mean'];axis='column' if overall<0 else 'row'
 qbar=feas['datasets'][name]['bbox_stats']['width_fraction' if axis=='column' else 'height_fraction']['mean']
 # Freeze the transfer rule exactly as specified: migrate the Aircraft
 # occupied-row fraction phi, using each dataset's mean bbox *height*.
 # Axis selection is a separate P5 decision and must not silently redefine b.
 hbar=feas['datasets'][name]['bbox_stats']['height_fraction']['mean'];b=.111/(3*(1-hbar))
 out['datasets'][name]={'axis':axis,'p5_column_minus_row':overall,'tall_p5_column_minus_row':tall,'wide_p5_column_minus_row':wide,
  'orientation_sign_flip':flip,'adaptive_feasibility_gain':gain,'adaptive_enabled':adaptive,'mean_selected_axis_box_fraction':qbar,
  'mean_bbox_height_fraction':hbar,'phi':.111,'background_min':b,'background_formula':'phi/(3*(1-mean bbox height fraction))'}
 if adaptive:raise RuntimeError(f'adaptive axis gate passed for {name}; this queue only implements pre-registered fixed-axis fallback')
Path(sys.argv[4]).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
PY

classes(){ [[ $1 == CUB_imsize224 ]]&&echo 200||echo 196;};batch(){ [[ $1 == CUB_imsize224 ]]&&echo 20||echo 14;}
mean(){ [[ $1 == CUB_imsize224 ]]&&echo '.4857 .4994 .4326'||echo '.4708 .4601 .4551';};std(){ [[ $1 == CUB_imsize224 ]]&&echo '.2260 .2215 .2595'||echo '.2885 .2879 .2962';}
for d in CUB_imsize224 SC_imsize224;do
 axis=$(python -c "import json;print(json.load(open('$EXP/audits/axis_decision.json'))['datasets']['$d']['axis'])")
 bg=$(python -c "import json;print(json.load(open('$EXP/audits/axis_decision.json'))['datasets']['$d']['background_min'])")
 read -ra mu<<<"$(mean "$d")";read -ra sigma<<<"$(std "$d")";dir="$STAGE/construction/$d";mkdir -p "$dir"
 python "$FG/prepare_fg_k3_bbox_pack.py" --selection-manifest "$RANDOM_ROOT/manifests/$d/rseed0/ipc10.json" \
  --bbox-manifest "$STAGE/bbox/${d}_bbox_manifest.json" --axis "$axis" --background-min "$bg" --output-root "$dir" \
  --mean "${mu[@]}" --std "${sigma[@]}" >"$EXP/logs/pack_${d}.log" 2>&1
 python "$FG/prepare_fg_public600_manifest.py" --bbox-manifest "$STAGE/bbox/${d}_bbox_manifest.json" \
  --selection-manifest "$RANDOM_ROOT/manifests/$d/rseed0/ipc10.json" --output "$STAGE/fit/${d}_public600.json" >"$EXP/logs/public600_${d}.log" 2>&1
done

CUDA_VISIBLE_DEVICES=0 python "$FG/fit_fg_fir_preemphasis.py" --public-manifest "$STAGE/fit/CUB_imsize224_public600.json" \
 --bbox-manifest "$STAGE/bbox/CUB_imsize224_bbox_manifest.json" --base-manifest "$STAGE/construction/CUB_imsize224/base_manifest.json" \
 --imagenet-weights "$WEIGHTS" --output-manifest "$STAGE/construction/CUB_imsize224/fir_manifest.json" --audit "$EXP/audits/fir_CUB.json" >"$EXP/logs/fit_CUB.log" 2>&1&p0=$!
CUDA_VISIBLE_DEVICES=1 python "$FG/fit_fg_fir_preemphasis.py" --public-manifest "$STAGE/fit/SC_imsize224_public600.json" \
 --bbox-manifest "$STAGE/bbox/SC_imsize224_bbox_manifest.json" --base-manifest "$STAGE/construction/SC_imsize224/base_manifest.json" \
 --imagenet-weights "$WEIGHTS" --output-manifest "$STAGE/construction/SC_imsize224/fir_manifest.json" --audit "$EXP/audits/fir_Cars.json" >"$EXP/logs/fit_Cars.log" 2>&1&p1=$!;wait $p0;wait $p1

python - "$STAGE" "$EXP/audits/construction.json" <<'PY'
import json,sys
from pathlib import Path
stage=Path(sys.argv[1]);rows=[]
for d,c in (('CUB_imsize224',200),('SC_imsize224',196)):
 for kind,file in (('bbox','base_manifest.json'),('f1','fir_manifest.json')):
  p=stage/'construction'/d/file;x=json.load(open(p));assert x['parents']==c*3 and x['unique_sources']==c*9 and len(x['records'])==c*3
  rows.append({'dataset':d,'kind':kind,'manifest':str(p),'axis':x['compression_axis'],'background_min':x['background_minimum'],'subject_density_stats':x['subject_density_stats']})
Path(sys.argv[2]).write_text(json.dumps({'status':'complete','rows':rows},indent=2)+'\n')
PY

relabel(){
 local d=$1 arm=$2 gpu=$3 c bs man mode source out;c=$(classes "$d");bs=$(batch "$d");man="$STAGE/construction/$d/base_manifest.json";mode=reference;source="$STAGE/fkd/$d/u9";out="$STAGE/fkd/$d/$arm"
 [[ $arm == bbox ]]&&mode=compressed
 if [[ $arm == f1 ]];then mode=compressed_fir;man="$STAGE/construction/$d/fir_manifest.json";fi
 [[ $arm == u9 ]]&&source="$out"
 CUDA_VISIBLE_DEVICES=$gpu python "$FG/relabel_aircraft_six_source_fir.py" --manifest "$man" --source-fkd "$source" --output-fkd "$out" \
  --teacher "$TEACHERS/$d/tseed42/ResNet18.pth" --image-mode "$mode" --batch-size "$bs" --classes "$c" --workers 6 >"$EXP/logs/relabel_${d}_${arm}.log" 2>&1
}
relabel CUB_imsize224 u9 0&p0=$!;relabel SC_imsize224 u9 1&p1=$!;wait $p0;wait $p1
for arm in bbox f1;do relabel CUB_imsize224 "$arm" 0&p0=$!;relabel SC_imsize224 "$arm" 1&p1=$!;wait $p0;wait $p1;done

train(){
 local d=$1 arm=$2 seed=$3 gpu=$4 bs man mode result out;bs=$(batch "$d");man="$STAGE/construction/$d/base_manifest.json";mode=$arm
 [[ $arm == f1 ]]&&{ man="$STAGE/construction/$d/fir_manifest.json";mode=compressed_fir;};[[ $arm == bbox ]]&&mode=compressed;[[ $arm == u9 ]]&&mode=reference
 result="$EXP/results/${d}_${arm}_sseed${seed}.json";out="$STAGE/post_eval/${d}_${arm}_sseed${seed}";[[ -f "$result" ]]&&return;mkdir -p "$out"
 CUDA_VISIBLE_DEVICES=$gpu python "$ROOT/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 3 --exp-name "transfer_${d}_${arm}_s${seed}" \
  --original-data-path "$STAGE/construction/$d/packed" --fkd-path "$STAGE/fkd/$d/$arm" --paired-source-manifest "$man" --paired-source-mode "$mode" \
  --output-dir "$out" --batch-size "$bs" --epochs 400 --dataset-name "$d" --gradient-accumulation-steps 2 --mix-type cutmix \
  --workers 6 --persistent-workers --fkd_seed 42 --train-seed "$seed" --temperature 20 --student-initialization imagenet-v1 \
  --student-protocol-name transfer_k3a1_soft_pff --adamw-weight-decay 1e-5 --adamw-beta1 .9 --adamw-beta2 .999 --adamw-eps 1e-8 \
  --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 --cosine-t-max 400 --cosine-eta-min 0 --val-dir "$DATA/$d/test" \
  --disable-wandb --per-class-output "$result" >"$EXP/logs/${d}_${arm}_sseed${seed}.log" 2>&1
}
jobs=();for d in CUB_imsize224 SC_imsize224;do for arm in u9 bbox f1;do for s in 42 43 44;do jobs+=("$d:$arm:$s");done;done;done
for((i=0;i<${#jobs[@]};i+=4));do pids=();for((j=0;j<4&&i+j<${#jobs[@]};j++));do IFS=: read -r d arm s<<<"${jobs[i+j]}";train "$d" "$arm" "$s" $((j%2))&pids+=("$!");done;failed=0;for p in "${pids[@]}";do wait "$p"||failed=1;done;((failed==0));done

python - "$EXP" <<'PY'
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);groups={};comparisons={}
def st(v):v=list(map(float,v));return {'mean':statistics.mean(v),'sample_sd':statistics.stdev(v),'values':v,'positive_count':sum(x>0 for x in v)}
for d in ('CUB_imsize224','SC_imsize224'):
 rows={}
 for arm in ('r0','u9','bbox','f1'):
  values={s:json.load(open(root/'results'/f'{d}_{arm}_sseed{s}.json')) for s in (42,43,44)};rows[arm]=values
  groups[f'{d}_{arm}']={'best':st(values[s]['best_top1'] for s in values),'final':st(values[s]['final_epoch_top1'] for s in values),'best_epochs':[values[s]['best_epoch'] for s in values]}
 def delta(a,b):return st(rows[a][s]['final_epoch_top1']-rows[b][s]['final_epoch_top1'] for s in (42,43,44))
 comparisons[d]={'k3a1_f1_minus_r0':delta('f1','r0'),'k3a1_f1_minus_u9':delta('f1','u9'),'bbox_minus_u9':delta('bbox','u9'),'f1_minus_bbox':delta('f1','bbox')}
out={'status':'complete','protocol':'cub_cars_k3a1_transfer_v1','groups':groups,'comparisons':comparisons,'axis_decision':json.load(open(root/'audits/axis_decision.json')),
 'new_fkd':8,'new_students':24,'main_phase_new_fkd':6,'main_phase_new_students':18}
(root/'summary'/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
PY
rm -f "$EXP/status/main.running";date --iso-8601=seconds>"$EXP/status/main.complete"
