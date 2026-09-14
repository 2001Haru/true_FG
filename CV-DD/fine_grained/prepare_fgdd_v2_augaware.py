"""FGDD-v2: score candidate replacement in exact CutMix/BSSL batch contexts."""
import argparse,hashlib,json,os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from prepare_fgdd_v1_aircraft import image_tensor,state_sha256

def stable(items,key):return sorted(items,key=lambda x:hashlib.sha256(f'{key}\0{x}'.encode()).digest())
def imagefolder(root):
 paths=[];targets=[];grouped=[]
 for c,d in enumerate(sorted(p for p in root.iterdir() if p.is_dir())):
  g=sorted(p.resolve() for p in d.iterdir() if p.is_file() or p.is_symlink());grouped.append(g);paths+=g;targets += [c]*len(g)
 return paths,torch.tensor(targets),grouped
def sampler_orders(n,epochs=400):
 gen=torch.Generator().manual_seed(42);out=[]
 for _ in range(epochs):out.append(torch.randperm(n,generator=gen).tolist());torch.randperm(n,generator=gen)[:0]
 return out
def reset_bn(model,state):
 modules=dict(model.named_modules())
 for name,values in state.items():
  module=modules[name]
  for field,value in values.items():getattr(module,field).copy_(value)
def bn_state(model):
 return {name:{field:getattr(module,field).detach().clone() for field in ('running_mean','running_var','num_batches_tracked')} for name,module in model.named_modules() if isinstance(module,nn.BatchNorm2d)}
def gradient(images,teacher,backbone,head,initial_bn,bbox,mix_index):
 x1,y1,x2,y2=map(int,bbox);mixed=images.clone();mixed[:,:,x1:x2,y1:y2]=images[mix_index,:,x1:x2,y1:y2]
 reset_bn(teacher,initial_bn);teacher.train()
 with torch.no_grad():
  split=len(mixed)//2;q=torch.cat((teacher(mixed[:split]),teacher(mixed[split:])),0).div(20).softmax(1)
  features=backbone(mixed)
 logits=head(features);p=logits.div(20).softmax(1);delta=(p-q)/20.0
 return torch.einsum('nk,nh->kh',delta,features)/len(images),delta.mean(0)
def class_grad(features,probs,c,exclude):
 keep=[i for i in range(len(features)) if i not in exclude];f=features[keep];p=probs[keep].clone();p[:,:,c]-=1
 return torch.einsum('nvk,nvh->kh',p,f)/(len(keep)*2),p.mean((0,1))
def main():
 p=argparse.ArgumentParser();p.add_argument('--train-root',required=True,type=Path);p.add_argument('--r0-root',required=True,type=Path);p.add_argument('--r2-root',required=True,type=Path);p.add_argument('--r0-fkd',required=True,type=Path);p.add_argument('--v1-root',required=True,type=Path);p.add_argument('--teacher',required=True,type=Path);p.add_argument('--output-root',required=True,type=Path);p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;a.output_root.mkdir(parents=True,exist_ok=True)
 full_paths,targets,full_grouped=imagefolder(a.train_root);path_index={x:i for i,x in enumerate(full_paths)}
 r0_paths,r0_targets,r0_grouped=imagefolder(a.r0_root);r2_paths,_,r2_grouped=imagefolder(a.r2_root)
 if len(r0_paths)!=300 or len(r2_paths)!=300:raise RuntimeError('expected IPC3 roots')
 cache=torch.load(a.v1_root/'selection/cache/imagenet_features.pt',map_location='cpu',weights_only=False);assert [str(x) for x in full_paths]==cache['paths'];features=cache['features'].float()
 head=nn.Linear(512,100);head.load_state_dict(torch.load(a.v1_root/'selection/cache/scorer_head.pth',map_location='cpu',weights_only=False));head.eval()
 with torch.no_grad():probs=head(features.reshape(-1,512)).reshape(len(full_paths),2,100).softmax(2)
 teacher=models.resnet18(weights=None);teacher.fc=nn.Linear(teacher.fc.in_features,100);teacher.load_state_dict(torch.load(a.teacher,map_location='cpu',weights_only=False));teacher.to(a.device);initial_bn=bn_state(teacher)
 backbone=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1);backbone.fc=nn.Identity();backbone.eval().to(a.device)
 for model in (teacher,backbone,head):
  for parameter in model.parameters():parameter.requires_grad_(False)
 head=head.to(a.device)
 orders=sampler_orders(300);r0_index={x:i for i,x in enumerate(r0_paths)}
 v1_manifest=json.loads((a.v1_root/'selection/selection_manifest.json').read_text());anchors=[];x0s=[];r2s=[];candidate_sets=[]
 for c in range(100):
  detail=v1_manifest['details'][c];common={Path(x).resolve() for x in detail['anchors']};x0=Path(detail['r0_third']).resolve();r2=Path(detail['r2']['path']).resolve()
  if len(common)!=2 or not common.issubset(r0_grouped[c]) or not common.issubset(r2_grouped[c]) or x0 not in r0_grouped[c] or r2 not in r2_grouped[c]:raise RuntimeError(f'class {c}: v1 role mismatch')
  anchors.append(sorted(common));x0s.append(x0);r2s.append(r2)
  pool=[x for x in full_grouped[c] if x not in common and x not in (x0,r2)];extra=stable(pool,f'fgdd-v2-candidates\0{c}')[:8 if x0!=r2 else 9];candidate_sets.append([x0]+([] if r2==x0 else [r2])+extra)
  if len(candidate_sets[-1])!=10 or len(set(candidate_sets[-1]))!=10:raise RuntimeError('candidate set error')
 class_means=[]
 for c in range(100):
  inds=[path_index[x] for x in full_grouped[c] if x not in anchors[c]];class_means.append(class_grad(features[inds],probs[inds],c,set()))
 details=[];selected=[];teacher_forward_images=0
 for c in range(100):
  other_w=torch.stack([class_means[k][0] for k in range(100) if k!=c]).mean(0)
  other_b=torch.stack([class_means[k][1] for k in range(100) if k!=c]).mean(0)
  references={}
  for candidate in candidate_sets[c]:
   excluded={x0s[c],candidate};own_indices=[path_index[x] for x in full_grouped[c] if x not in anchors[c] and x not in excluded]
   own_w,own_b=class_grad(features[own_indices],probs[own_indices],c,set())
   references[candidate]=(0.5*own_w+0.5*other_w,0.5*own_b+0.5*other_b)
  epochs=sorted(stable(range(400),f'fgdd-v2-contexts\0{c}')[:8]);scores={x:0.0 for x in candidate_sets[c]};context_scores={x:[] for x in candidate_sets[c]};contexts=[]
  for epoch in epochs:
   order=orders[epoch];position=order.index(r0_index[x0s[c]]);batch=position//20;row=position%20;batch_indices=order[batch*20:(batch+1)*20]
   payload=torch.load(a.r0_fkd/f'epoch_{epoch}/batch_{batch}.tar',map_location='cpu',weights_only=False);coords,flips,mix_index,_,bbox=payload[:5]
   if mix_index is None or len(batch_indices)!=20:raise RuntimeError('invalid scoring context')
   expected=torch.tensor([0.,0.,1.,1.]);
   if not all(torch.allclose(torch.as_tensor(x).float(),expected,atol=1e-7,rtol=0) for x in coords):raise RuntimeError('context is not full-frame')
   base=torch.stack([image_tensor(r0_paths[index],bool(flips[i])) for i,index in enumerate(batch_indices)]).to(a.device)
   original_w,original_b=gradient(base,teacher,backbone,head,initial_bn,bbox,mix_index.to(a.device));teacher_forward_images+=20
   context_scores[x0s[c]].append(0.0)
   for candidate in candidate_sets[c]:
    if candidate==x0s[c]:continue
    changed=base.clone();changed[row]=image_tensor(candidate,bool(flips[row])).to(a.device)
    cand_w,cand_b=gradient(changed,teacher,backbone,head,initial_bn,bbox,mix_index.to(a.device));teacher_forward_images+=20
    ref_w,ref_b=references[candidate];contribution=float(((cand_w-original_w).cpu()*ref_w).sum()+((cand_b-original_b).cpu()*ref_b).sum());scores[candidate]+=contribution;context_scores[candidate].append(contribution)
   contexts.append({'epoch':epoch,'batch':batch,'row':row,'bbox':list(map(int,bbox)),'flip':bool(flips[row]),'original_donor_uses':int((mix_index==row).sum())})
  scores={path:value/8.0 for path,value in scores.items()};choice=max(candidate_sets[c],key=lambda x:(scores[x],tuple(-b for b in x.as_posix().encode())));selected.append(choice)
  details.append({'class_id':c,'anchors':[str(x) for x in anchors[c]],'r0_third':str(x0s[c]),'r2_third':str(r2s[c]),'candidates':[{'path':str(x),'score':scores[x],'context_scores':context_scores[x]} for x in candidate_sets[c]],'selected':str(choice),'selected_score':scores[choice],'contexts':contexts})
 root=a.output_root/'selected/r2_augaware/ipc3'
 for c in range(100):
  directory=root/f'{c:03d}';directory.mkdir(parents=True,exist_ok=True);chosen=anchors[c]+[selected[c]]
  if len(set(chosen))!=3:raise RuntimeError('duplicate output')
  for source in sorted(chosen):
   destination=directory/source.name
   if destination.exists() or destination.is_symlink():
    if destination.resolve()!=source:raise RuntimeError(f'collision {destination}')
   else:os.symlink(source,destination)
 overlap={'selected_is_r0':sum(x==y for x,y in zip(selected,x0s)),'selected_is_r2':sum(x==y for x,y in zip(selected,r2s))}
 manifest={'status':'complete','dataset':'A_imsize224','ipc':3,'method':'r2_augaware_batch_replacement','candidate_count_per_class':10,'contexts_per_candidate':8,'candidate_rule':'R0 third, R2 third, stable-SHA random distinct fill','context_rule':'8 stable-SHA epochs/class from R0 fullframe+CutMix FKD; replace R0 third at its batch row','score':'mean_context dot(g_ref_R2, g_candidate_batch-g_R0_batch)','reference':'R2: 50% own class + 50% class-balanced other99; exclude candidate and replaced R0 image','scorer':'exact v1 frozen ImageNet-V1 ResNet18 eval backbone and trained linear head','scorer_head_sha256':state_sha256(torch.load(a.v1_root/'selection/cache/scorer_head.pth',map_location='cpu',weights_only=False)),'teacher':str(a.teacher.resolve()),'teacher_mode':'train BSSL, BN buffers restored before every batch comparison','training_view':'fullframe+flip+CutMix replay, actual mixed-view Teacher T20 KL, no T2','teacher_forward_images':teacher_forward_images,'expected_teacher_forward_images':160000,'overlap':overlap,'details':details}
 if teacher_forward_images!=160000:raise RuntimeError(f'query budget mismatch {teacher_forward_images}')
 output=a.output_root/'selection_manifest.json';tmp=output.with_suffix('.json.tmp');tmp.parent.mkdir(parents=True,exist_ok=True);tmp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');os.replace(tmp,output);print(json.dumps({'status':'complete','queries':teacher_forward_images,'overlap':overlap},indent=2))
if __name__=='__main__':main()
