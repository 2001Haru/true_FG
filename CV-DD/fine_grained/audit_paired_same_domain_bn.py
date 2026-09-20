"""Recompute BN using the exact paired-source training image domain and trajectory."""
import argparse,glob,json,statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,models,transforms
from paired_source_dataset import PairedSourceIndex,AIRCRAFT_MEAN,AIRCRAFT_STD
def st(x):x=list(map(float,x));return {'mean':statistics.mean(x),'sample_sd':statistics.stdev(x),'values':x}
def load(path,dev):
 s=torch.load(path,map_location='cpu',weights_only=False)['state_dict'];s={k.removeprefix('module.'):v for k,v in s.items()};m=models.resnet18(weights=None);m.fc=nn.Linear(m.fc.in_features,100);m.load_state_dict(s);return m.to(dev)
@torch.no_grad()
def evaluate(m,loader,dev):
 m.eval();n=c=0;loss=0.;ce=nn.CrossEntropyLoss(reduction='sum')
 for x,y in loader:x=x.to(dev);y=y.to(dev);z=m(x);loss+=ce(z,y).item();c+=z.argmax(1).eq(y).sum().item();n+=len(y)
 return {'top1':100*c/n,'nll':loss/n}
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,type=Path);p.add_argument('--mode',required=True);p.add_argument('--fkd',required=True,type=Path);p.add_argument('--checkpoints',required=True);p.add_argument('--test-root',required=True,type=Path);p.add_argument('--epochs',type=int,default=40);p.add_argument('--device',default='cuda:0');p.add_argument('--disable-cutmix',action='store_true');p.add_argument('--output',required=True,type=Path);a=p.parse_args();torch.set_num_threads(1);dev=torch.device(a.device);index=PairedSourceIndex(a.manifest,a.mode)
 def get(z):
  parent,source=z;im=index.load(parent,source);x=transforms.functional.pil_to_tensor(im).float().div_(255);return transforms.functional.normalize(x,AIRCRAFT_MEAN,AIRCRAFT_STD).half()
 with ThreadPoolExecutor(max_workers=16) as pool:base=torch.stack(list(pool.map(get,[(p,s) for p in range(300) for s in (0,1)]))).reshape(300,2,3,224,224)
 configs=[]
 for e in range(a.epochs):
  for b in range(15):
   c=torch.load(a.fkd/f'epoch_{e}/batch_{b}.tar',map_location='cpu',weights_only=False);configs.append((c[1].bool(),c[2],c[3],c[4],c[6].long(),c[7].long()))
 test=datasets.ImageFolder(a.test_root,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(AIRCRAFT_MEAN,AIRCRAFT_STD)]));loader=DataLoader(test,batch_size=256,num_workers=6,persistent_workers=True,pin_memory=True);rows=[]
 for path in map(Path,sorted(glob.glob(a.checkpoints))):
  m=load(path,dev);m.requires_grad_(False);params={n:v.cpu().clone() for n,v in m.named_parameters()};before=evaluate(m,loader,dev)
  m.eval()
  for z in m.modules():
   if isinstance(z,nn.BatchNorm2d):z.reset_running_stats();z.momentum=None;z.train()
  with torch.no_grad():
   for flip,mi,ml,mb,sources,parents in configs:
    x=base[parents,sources].clone();x[flip]=torch.flip(x[flip],(-1,));x=x.to(dev,dtype=torch.float32)
    if not a.disable_cutmix:
     # Exact replay of utils_fkd.cutmix in fkd_load mode, kept local so this
     # diagnostic does not import unrelated model definitions.
     rand_index=mi.to(dev);bbx1,bby1,bbx2,bby2=mb
     x[:,:,bbx1:bbx2,bby1:bby2]=x[rand_index,:,bbx1:bbx2,bby1:bby2]
    m(x[:10]);m(x[10:])
  after=evaluate(m,loader,dev);rows.append({'checkpoint':str(path),'before':before,'after':after,'gain_top1':after['top1']-before['top1'],'gain_nll':after['nll']-before['nll'],'parameter_max_abs_change':max(float((v.cpu()-params[n]).abs().max()) for n,v in m.named_parameters())})
 out={'status':'complete','protocol':'paired_same_domain_bn_v1','mode':a.mode,'epochs':a.epochs,'views':a.epochs*300,'cutmix':not a.disable_cutmix,'rows':rows,'summary':{'before_top1':st(r['before']['top1'] for r in rows),'after_top1':st(r['after']['top1'] for r in rows),'gain_top1':st(r['gain_top1'] for r in rows),'gain_nll':st(r['gain_nll'] for r in rows)}};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['summary'],indent=2))
if __name__=='__main__':main()
