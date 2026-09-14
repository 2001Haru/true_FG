import argparse,json,os
from pathlib import Path
import torch
def equal(a,b):
 if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
def main():
 p=argparse.ArgumentParser();p.add_argument('--reference',required=True,type=Path);p.add_argument('--candidate',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--epochs',type=int,default=400);p.add_argument('--batches',type=int,default=15);a=p.parse_args();names=('coords','flip','mix_index','mix_lam','mix_bbox');mismatch={x:0 for x in names}
 for e in range(a.epochs):
  for b in range(a.batches):
   x=torch.load(a.reference/f'epoch_{e}/batch_{b}.tar',map_location='cpu',weights_only=False);y=torch.load(a.candidate/f'epoch_{e}/batch_{b}.tar',map_location='cpu',weights_only=False)
   if len(x)!=6 or len(y)!=6:raise RuntimeError('payload length')
   for i,name in enumerate(names):mismatch[name]+=not equal(x[i],y[i])
 result={'status':'complete' if sum(mismatch.values())==0 else 'failed','reference':str(a.reference.resolve()),'candidate':str(a.candidate.resolve()),'epochs':a.epochs,'batches_per_epoch':a.batches,'compared_batches':a.epochs*a.batches,'mismatch_batches':mismatch,'total_mismatches':sum(mismatch.values())};a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');os.replace(tmp,a.output);print(json.dumps(result));
 if result['status']!='complete':raise SystemExit(1)
if __name__=='__main__':main()
