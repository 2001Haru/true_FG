"""Replay R0 FKD views with half-random, half graph-guided one-to-one CutMix pairing."""
import argparse,hashlib,json,os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from prepare_fgdd_v1_aircraft import image_tensor

def imagefolder(root):
 paths=[];targets=[]
 for c,d in enumerate(sorted(p for p in root.iterdir() if p.is_dir())):
  g=sorted(p.resolve() for p in d.iterdir() if p.is_file() or p.is_symlink());paths+=g;targets += [c]*len(g)
 return paths,targets
def stable(items,key):return sorted(items,key=lambda x:hashlib.sha256(f'{key}\0{x}'.encode()).digest())
def derangement(seed,n=100):
 g=torch.Generator().manual_seed(seed)
 while True:
  p=torch.randperm(n,generator=g)
  if not bool((p==torch.arange(n)).any()):return p.tolist()
def relabel_graph(neighbors,permutation):
 inverse=[0]*len(permutation)
 for old,new in enumerate(permutation):inverse[new]=old
 return [[permutation[k] for k in neighbors[inverse[c]]] for c in range(len(permutation))]
def guided_permutation(original,classes,neighbors,key):
 guided=set(stable(range(len(classes)),f'{key}\0guided_receivers')[:len(classes)//2]);retained=[r for r in range(len(classes)) if r not in guided];used={int(original[r]) for r in retained};available=set(range(len(classes)))-used
 options={r:[d for d in available if classes[d] in neighbors[classes[r]]] for r in guided};receiver_order=stable(guided,f'{key}\0matching');receiver_order.sort(key=lambda r:len(options[r]))
 donor_to_receiver={}
 def visit(receiver,seen):
  for donor in stable(options[receiver],f'{key}\0receiver{receiver}'):
   if donor in seen:continue
   seen.add(donor)
   if donor not in donor_to_receiver or visit(donor_to_receiver[donor],seen):donor_to_receiver[donor]=receiver;return True
  return False
 for receiver in receiver_order:visit(receiver,set())
 result=[None]*len(classes)
 for r in retained:result[r]=int(original[r])
 for donor,receiver in donor_to_receiver.items():result[receiver]=donor
 unmatched_r=[r for r in guided if result[r] is None];unmatched_d=list(available-set(donor_to_receiver))
 unmatched_r=stable(unmatched_r,f'{key}\0fallback_receivers');unmatched_d=stable(unmatched_d,f'{key}\0fallback_donors')
 for receiver,donor in zip(unmatched_r,unmatched_d):result[receiver]=donor
 if sorted(result)!=list(range(len(classes))) or sum(result[r]==int(original[r]) for r in retained)!=len(retained):raise RuntimeError('pairing invariant failed')
 hits=sum(classes[result[r]] in neighbors[classes[r]] for r in guided)
 return torch.tensor(result),{'retained_random_receivers':len(retained),'guided_receivers':len(guided),'guided_neighbor_hits':hits,'guided_fallbacks':len(guided)-hits}
def main():
 p=argparse.ArgumentParser();p.add_argument('--images',required=True,type=Path);p.add_argument('--source-fkd',required=True,type=Path);p.add_argument('--destination',required=True,type=Path);p.add_argument('--selection-manifest',required=True,type=Path);p.add_argument('--teacher',required=True,type=Path);p.add_argument('--graph',required=True,choices=('confusion','randomized'));p.add_argument('--random-graph-seed',type=int,default=20260914);p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 paths,targets=imagefolder(a.images)
 if len(paths)!=300:raise RuntimeError('expected 300 images')
 manifest=json.loads(a.selection_manifest.read_text());neighbors=[row['neighbors'] for row in manifest['details']];class_permutation=None
 if a.graph=='randomized':class_permutation=derangement(a.random_graph_seed);neighbors=relabel_graph(neighbors,class_permutation)
 teacher=models.resnet18(weights=None);teacher.fc=nn.Linear(teacher.fc.in_features,100);teacher.load_state_dict(torch.load(a.teacher,map_location='cpu',weights_only=False));teacher.train().to(a.device)
 for parameter in teacher.parameters():parameter.requires_grad_(False)
 generator=torch.Generator().manual_seed(42);a.destination.mkdir(parents=True,exist_ok=True);stats={'batches':0,'views':0,'retained_random_receivers':0,'guided_receivers':0,'guided_neighbor_hits':0,'guided_fallbacks':0};max_donor_count=0
 with torch.no_grad():
  for epoch in range(400):
   order=torch.randperm(300,generator=generator).tolist();torch.randperm(300,generator=generator)[:0];out=a.destination/f'epoch_{epoch}';out.mkdir(exist_ok=True)
   for batch,offset in enumerate(range(0,300,20)):
    payload=torch.load(a.source_fkd/f'epoch_{epoch}/batch_{batch}.tar',map_location='cpu',weights_only=False);coords,flips,original,lam,bbox=payload[:5];indices=order[offset:offset+20];batch_classes=[targets[i] for i in indices]
    pairing,s=guided_permutation(original,batch_classes,neighbors,f'fgdd-v2-pairing\0{a.graph}\0{epoch}\0{batch}')
    images=torch.stack([image_tensor(paths[index],bool(flips[row])) for row,index in enumerate(indices)]).to(a.device);x1,y1,x2,y2=map(int,bbox);mixed=images.clone();pairing_gpu=pairing.to(a.device);mixed[:,:,x1:x2,y1:y2]=images[pairing_gpu,:,x1:x2,y1:y2]
    split=len(mixed)//2;logits=torch.cat((teacher(mixed[:split]),teacher(mixed[split:])),0).half().cpu();payload[2]=pairing;payload[5]=logits;torch.save(payload,out/f'batch_{batch}.tar')
    for key,value in s.items():stats[key]+=value
    stats['batches']+=1;stats['views']+=20;max_donor_count=max(max_donor_count,max(torch.bincount(pairing,minlength=20).tolist()))
 result={'status':'complete','graph':a.graph,'source_fkd':str(a.source_fkd.resolve()),'destination':str(a.destination.resolve()),'images':str(a.images.resolve()),'teacher':str(a.teacher.resolve()),'teacher_mode':'train BSSL','epochs':400,'batch_size':20,'random_receiver_fraction':0.5,'guided_receiver_fraction':0.5,'pairing':'retain original donors for 10 stable-hash receivers; maximum-cardinality neighbor matching for other 10; stable-hash one-to-one fallback','class_permutation':class_permutation,'random_graph_seed':a.random_graph_seed if a.graph=='randomized' else None,'stats':stats,'guided_hit_rate':stats['guided_neighbor_hits']/stats['guided_receivers'],'max_donor_multiplicity':max_donor_count,'teacher_forward_images':stats['views'],'metadata_replay':['batch members','BSSL microbatch grouping','flip','bbox','CutMix trigger'],'logits':'Teacher prediction on actual mixed view, fp16 raw 100-class'}
 output=a.destination/'pairing_manifest.json';tmp=output.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');os.replace(tmp,output);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
