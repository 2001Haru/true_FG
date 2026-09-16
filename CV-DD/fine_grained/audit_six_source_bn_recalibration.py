"""Post-hoc BN recalibration of seed0 U6/BBox/object-decoded final Students."""
import argparse,json,statistics
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,models,transforms
from paired_source_dataset import AIRCRAFT_MEAN,AIRCRAFT_STD,PairedSourceIndex
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from relabel.utils_fkd import mix_aug
def st(v):v=list(map(float,v));return {'mean':statistics.mean(v),'sample_sd':statistics.stdev(v),'values':v}
def ranks(x):
 order=np.argsort(x,kind='mergesort');out=np.empty(len(x),float);i=0
 while i<len(x):
  j=i+1
  while j<len(x) and x[order[j]]==x[order[i]]:j+=1
  out[order[i:j]]=(i+j-1)/2+1;i=j
 return out
def spearman(x,y):return float(np.corrcoef(ranks(np.asarray(x)),ranks(np.asarray(y)))[0,1])
@torch.no_grad()
def evaluate(model,loader):
 model.eval();correct=total=0;nll=0.;per_c=torch.zeros(100,dtype=torch.long);per_n=torch.zeros(100,dtype=torch.long);ce=nn.CrossEntropyLoss(reduction='sum')
 for images,target in loader:
  images=images.cuda(non_blocking=True);target=target.cuda(non_blocking=True);logits=model(images);nll+=ce(logits,target).item();pred=logits.argmax(1);correct+=pred.eq(target).sum().item();total+=len(target);per_n.scatter_add_(0,target.cpu(),torch.ones_like(target.cpu()));per_c.scatter_add_(0,target.cpu(),pred.eq(target).cpu().long())
 return {'top1':100*correct/total,'nll':nll/total,'per_class':[100*int(per_c[i])/max(1,int(per_n[i])) for i in range(100)]}
def load_model(path):
 x=torch.load(path,map_location='cpu',weights_only=False);state=x['state_dict'];state={k.removeprefix('module.'):v for k,v in state.items()};m=models.resnet18(weights=None);m.fc=nn.Linear(m.fc.in_features,100);m.load_state_dict(state);return m.cuda()
@torch.no_grad()
def recalibrate(model,index,fkd,epochs,checkpoints):
 for m in model.modules():
  if isinstance(m,nn.BatchNorm2d):m.reset_running_stats();m.momentum=None
 model.train();mixargs=SimpleNamespace(mode='fkd_load',mix_type='cutmix',cutmix=1.,mixup=.8);snap={}
 for epoch in range(epochs):
  for batch in range(15):
   c=torch.load(fkd/f'epoch_{epoch}/batch_{batch}.tar',map_location='cpu',weights_only=False);flip,mix_index,mix_lam,mix_bbox,sources,parents=c[1],c[2],c[3],c[4],c[6],c[7];images=[]
   for parent,source,do_flip in zip(parents.tolist(),sources.tolist(),flip.tolist()):
    image=index.load(parent,source)
    if do_flip:image=transforms.functional.hflip(image)
    images.append(transforms.functional.normalize(transforms.functional.pil_to_tensor(image).float().div_(255),AIRCRAFT_MEAN,AIRCRAFT_STD))
   images=torch.stack(images).cuda();mixed,_,_,_=mix_aug(images,mixargs,mix_index,mix_lam,mix_bbox);model(mixed[:10]);model(mixed[10:])
  if epoch+1 in checkpoints:snap[epoch+1]={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k}
 return snap
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,type=Path);p.add_argument('--source-fkd',required=True,type=Path);p.add_argument('--test-root',required=True,type=Path);p.add_argument('--parent',required=True,type=Path);p.add_argument('--loss-root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--epochs',type=int,default=40);a=p.parse_args()
 test=datasets.ImageFolder(a.test_root,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(AIRCRAFT_MEAN,AIRCRAFT_STD)]));loader=DataLoader(test,batch_size=256,shuffle=False,num_workers=8,persistent_workers=True,pin_memory=True);index=PairedSourceIndex(a.manifest,'reference');groups={}
 specs={'reference':(a.parent/'post_eval/soft/reference','six_source_soft_reference',a.parent/'results/soft/reference'),'compressed':(a.parent/'post_eval/soft/compressed','six_source_soft_compressed',a.parent/'results/soft/compressed'),'object_decoded':(a.loss_root/'post_eval/object_decoded','six_source_object_decoded',a.loss_root/'results/object_decoded')}
 for group,(root,name,result_root) in specs.items():
  groups[group]={}
  for seed in (42,43,44):
   ckpt=root/f'sseed{seed}/A_imsize224/{name}_s{seed}/checkpoint.pth.tar';model=load_model(ckpt);before=evaluate(model,loader);expected=json.loads((result_root/f'sseed{seed}.json').read_text())['final_epoch_top1']
   if abs(before['top1']-expected)>1e-6:raise RuntimeError(f'baseline mismatch {group} {seed}: {before["top1"]} {expected}')
   original={n:p.detach().cpu().clone() for n,p in model.named_parameters()};snap=recalibrate(model,index,a.source_fkd,a.epochs,(20,40));after40=evaluate(model,loader)
   model.load_state_dict({**model.state_dict(),**snap[20]});after20=evaluate(model,loader)
   max_param=max(float((p.detach().cpu()-original[n]).abs().max()) for n,p in model.named_parameters())
   groups[group][str(seed)]={'before':before,'after20':after20,'after40':after40,'gain20':after20['top1']-before['top1'],'gain40':after40['top1']-before['top1'],'parameter_max_abs_change':max_param}
 summary={}
 for g in groups:summary[g]={'gain20':st(groups[g][str(s)]['gain20'] for s in (42,43,44)),'gain40':st(groups[g][str(s)]['gain40'] for s in (42,43,44)),'before_top1':st(groups[g][str(s)]['before']['top1'] for s in (42,43,44)),'after40_top1':st(groups[g][str(s)]['after40']['top1'] for s in (42,43,44))}
 for g in ('compressed','object_decoded'):summary[g+'_minus_reference_gain40']=st(groups[g][str(s)]['gain40']-groups['reference'][str(s)]['gain40'] for s in (42,43,44))
 manifest=json.loads(a.manifest.read_text());gap=[];decline=[]
 for c in range(100):
  sources=[z for row in manifest['records'] if row['class_id']==c for z in row['sources']];gap.append(sum(z['bbox_rows']*(1-max(z['density'])) for z in sources)/sum(z['bbox_rows'] for z in sources));decline.append(statistics.mean(groups['reference'][str(s)]['before']['per_class'][c]-groups['compressed'][str(s)]['before']['per_class'][c] for s in (42,43,44)))
 rng=np.random.default_rng(20260916);boot=[]
 for _ in range(10000):idx=rng.integers(0,100,100);boot.append(spearman(np.asarray(gap)[idx],np.asarray(decline)[idx]))
 attribution={'accuracy':'final-checkpoint per-class official-test accuracy','spearman_rho':spearman(gap,decline),'student_rhos':[spearman(gap,[groups['reference'][str(s)]['before']['per_class'][c]-groups['compressed'][str(s)]['before']['per_class'][c] for c in range(100)]) for s in (42,43,44)],'bootstrap95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]}
 result={'status':'complete','protocol':'six_source_bn_recalibration_diagnostic_v1','recalibration':'reset all BN running stats; cumulative momentum=None; replay first40 epochs of clear U6 actual fullframe+flip+CutMix views; forward microbatches10; evaluate at20/40 epochs','summary':summary,'final_per_class_density_attribution':attribution,'groups':groups}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'complete','summary':summary},indent=2))
if __name__=='__main__':main()
