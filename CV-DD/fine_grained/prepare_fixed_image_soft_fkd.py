"""Generate fixed per-image BSSL probabilities and area-mixed FKD caches."""
import argparse,hashlib,json,os
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torchvision import models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

MEAN=(0.4865,0.5177,0.5425);STD=(0.2124,0.2051,0.2375)
EXT={'.jpg','.jpeg','.png','.bmp','.ppm','.pgm','.tif','.tiff','.webp'}
def sha(path):
 d=hashlib.sha256()
 with path.open('rb') as h:
  for b in iter(lambda:h.read(1048576),b''):d.update(b)
 return d.hexdigest()
def tensor(path,flip):
 with Image.open(path) as im:
  im=TF.resize(im.convert('RGB'),[224,224],interpolation=InterpolationMode.BILINEAR)
  if flip:im=TF.hflip(im)
  return TF.normalize(TF.to_tensor(im),MEAN,STD)
class Views(Dataset):
 def __init__(self,items):self.items=items
 def __len__(self):return len(self.items)
 def __getitem__(self,i):
  index,path,view=self.items[i];return tensor(path,bool(view)),index,view
def classes(root):
 out=[]
 for directory in sorted(p for p in root.iterdir() if p.is_dir()):
  out.append(sorted(p.resolve() for p in directory.iterdir() if p.is_file() or p.is_symlink()))
 return out
def interleave(grouped):
 out=[]
 for rank in range(max(map(len,grouped))):
  for group in grouped:
   if rank<len(group):out.append(group[rank])
 return out
def fixed(args):
 unions=[set() for _ in range(100)]
 roots_grouped=[]
 for root in args.image_roots:
  current=classes(root)
  if len(current)!=100 or any(len(x)!=3 for x in current):raise RuntimeError(f'invalid IPC3 root {root}')
  roots_grouped.append(current)
  for c in range(100):unions[c].update(current[c])
 grouped=[sorted(x) for x in unions];paths=[p for group in grouped for p in group];index={p:i for i,p in enumerate(paths)}
 grouped_indices=[[index[p] for p in group] for group in grouped]
 order=interleave(grouped_indices);items=[(i,paths[i],v) for v in (0,1) for i in order]
 model=models.resnet18(weights=None);model.fc=nn.Linear(model.fc.in_features,100);model.load_state_dict(torch.load(args.teacher,map_location='cpu',weights_only=False),strict=True);model.train().to(args.device)
 for p in model.parameters():p.requires_grad_(False)
 loader=DataLoader(Views(items),batch_size=20,shuffle=False,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0)
 probs=torch.empty(len(paths),2,100,dtype=torch.float32);batches=0
 with torch.no_grad():
  for images,indices,views in loader:
   images=images.to(args.device,non_blocking=True);split=images.shape[0]//2
   logits=torch.cat((model(images[:split]),model(images[split:])),0)
   q=(logits/20.0).softmax(1).float().cpu();probs[indices.long(),views.long()]=q;batches+=1
 if not torch.allclose(probs.sum(2),torch.ones(len(paths),2),rtol=0,atol=2e-6):raise RuntimeError('fixed probabilities do not sum to one')
 args.labels.parent.mkdir(parents=True,exist_ok=True);torch.save({'paths':[str(p) for p in paths],'probabilities':probs},args.labels)
 context={'status':'complete','teacher':str(args.teacher.resolve()),'teacher_sha256':sha(args.teacher),'teacher_mode':'train','temperature':20.0,'probability_dtype':'float32','image_roots':[str(p.resolve()) for p in args.image_roots],'unique_images':len(paths),'shared_images':sum(len(roots_grouped[0][c]) + len(roots_grouped[1][c]) - len(grouped[c]) for c in range(100)),'views':len(items),'queries':len(items),'batch_size':20,'forward_split':'floor(n/2)+remainder','batches':batches,'order_sha256':hashlib.sha256('\n'.join(f'{v}\t{paths[i]}' for i,_,v in items).encode()).hexdigest(),'view_transform':'RGB bilinear Resize224; original or horizontal flip; Aircraft normalization','mix':None}
 args.context.write_text(json.dumps(context,indent=2,sort_keys=True)+'\n');print(json.dumps(context,indent=2))
def derive(args):
 labels=torch.load(args.labels,map_location='cpu',weights_only=False);lookup={Path(p).resolve():labels['probabilities'][i] for i,p in enumerate(labels['paths'])}
 grouped=classes(args.images);paths=[p for group in grouped for p in group]
 if len(paths)!=300:raise RuntimeError('expected 300 images')
 generator=torch.Generator().manual_seed(42);args.destination.mkdir(parents=True,exist_ok=True);views=0;max_sum=0.0
 for epoch in range(400):
  order=torch.randperm(300,generator=generator).tolist();torch.randperm(300,generator=generator)[:0]
  out=args.destination/f'epoch_{epoch}';out.mkdir(exist_ok=True)
  for batch,offset in enumerate(range(0,300,20)):
   payload=torch.load(args.source/f'epoch_{epoch}/batch_{batch}.tar',map_location='cpu',weights_only=False)
   coords,flips,mix_index,_,bbox,_=payload[:6]
   if mix_index is None or bbox is None:raise RuntimeError('source is not CutMix FKD')
   indices=order[offset:offset+len(flips)];base=torch.stack([lookup[paths[index]][int(bool(flips[row]))] for row,index in enumerate(indices)])
   x1,y1,x2,y2=map(int,bbox);lam=1.0-((x2-x1)*(y2-y1))/(224.0*224.0)
   mixed=lam*base+(1.0-lam)*base[mix_index.long()]
   max_sum=max(max_sum,float((mixed.sum(1)-1).abs().max()));payload[5]=mixed
   torch.save(payload,out/f'batch_{batch}.tar');views+=len(indices)
 manifest={'status':'complete','source_fkd':str(args.source.resolve()),'destination_fkd':str(args.destination.resolve()),'images':str(args.images.resolve()),'fixed_labels':str(args.labels.resolve()),'fixed_context':str(args.context.resolve()),'epochs':400,'views':views,'target':'actual-bbox-area mixture of two fixed T20 probability vectors selected by replayed flip','student_view_metadata':'exact source batch order, full-frame coords, flip, CutMix index and bbox','max_probability_sum_error':max_sum,'teacher_queries_for_derivation':0}
 (args.destination/'fixed_target_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');print(json.dumps(manifest,indent=2))
def main():
 torch.set_num_threads(4);p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
 a=sub.add_parser('fixed');a.add_argument('--image-roots',required=True,type=Path,nargs=2);a.add_argument('--teacher',required=True,type=Path);a.add_argument('--labels',required=True,type=Path);a.add_argument('--context',required=True,type=Path);a.add_argument('--workers',type=int,default=8);a.add_argument('--device',default='cuda')
 b=sub.add_parser('derive');b.add_argument('--source',required=True,type=Path);b.add_argument('--destination',required=True,type=Path);b.add_argument('--images',required=True,type=Path);b.add_argument('--labels',required=True,type=Path);b.add_argument('--context',required=True,type=Path)
 args=p.parse_args();fixed(args) if args.command=='fixed' else derive(args)
if __name__=='__main__':main()
