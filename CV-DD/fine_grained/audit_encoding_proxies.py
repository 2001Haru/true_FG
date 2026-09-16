"""Evaluate training-free distortion proxies against ten observed Student losses."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models,transforms
from paired_source_dataset import PairedSourceIndex
from audit_six_source_feasibility_attribution import spearman
TMEAN=(.4865,.5177,.5425);TSTD=(.2124,.2051,.2375);IMEAN=(.485,.456,.406);ISTD=(.229,.224,.225)
LOSSES={'single':{'isotropic158':6.0706070607,'vertical112':4.2504250425,'horizontal112':12.2012201220,'bbox_retarget112':3.2903290329,'attention_retarget112':4.4604460446},'six':{'uniform':3.3903390339,'bbox':2.2902290229,'background_hsmooth':2.3202320232,'object_decoded':1.8801880188,'background_decoded':.4600460046}}
def load_resnet(path,classes):
 x=torch.load(path,map_location='cpu',weights_only=False);state=x.get('state_dict',x.get('model',x));state={k.removeprefix('module.'):v for k,v in state.items()};m=models.resnet18(weights=None);m.fc=nn.Linear(m.fc.in_features,classes);m.load_state_dict(state);return m.cuda().eval()
def normalize(x,mean,std):return (x-torch.tensor(mean,device=x.device)[None,:,None,None])/torch.tensor(std,device=x.device)[None,:,None,None]
def kl20(a,b):
 loga=F.log_softmax(a/20,1);logb=F.log_softmax(b/20,1);pa=loga.exp();return (pa*(loga-logb)).sum(1)
def feature_distance(a,b,boxes):
 out=[];h=a.shape[2]
 for i,(y0,y1) in enumerate(boxes):
  lo=max(0,min(h-1,math.floor(y0/224*h)));hi=max(lo+1,min(h,math.ceil(y1/224*h)));x=F.normalize(a[i,:,lo:hi].reshape(1,-1),dim=1);y=F.normalize(b[i,:,lo:hi].reshape(1,-1),dim=1);out.append(float((1-(x*y).sum()).cpu()))
 return out
def ssim_loss(x,y):
 size=11;sigma=1.5;v=torch.arange(size,device=x.device)-size//2;g=torch.exp(-(v.float()**2)/(2*sigma*sigma));g/=g.sum();w=(g[:,None]*g[None,:])[None,None].repeat(3,1,1,1)
 mu1=F.conv2d(x,w,padding=size//2,groups=3);mu2=F.conv2d(y,w,padding=size//2,groups=3);s1=F.conv2d(x*x,w,padding=size//2,groups=3)-mu1*mu1;s2=F.conv2d(y*y,w,padding=size//2,groups=3)-mu2*mu2;s12=F.conv2d(x*y,w,padding=size//2,groups=3)-mu1*mu2
 return 1-(((2*mu1*mu2+.01**2)*(2*s12+.03**2))/((mu1*mu1+mu2*mu2+.01**2)*(s1+s2+.03**2))).mean((1,2,3))
def tensor(image):return transforms.functional.pil_to_tensor(image).float().div_(255)
def model_pair(model,o,v,mean,std,features=False):
 cache={};handles=[]
 if features:
  for name in ('layer2','layer3'):handles.append(getattr(model,name).register_forward_hook(lambda m,i,out,n=name:cache.__setitem__(n,out.detach())))
 z=torch.cat((normalize(o,mean,std),normalize(v,mean,std)),0);logits=model(z);n=len(o);acts={k:(val[:n],val[n:]) for k,val in cache.items()}
 for h in handles:h.remove()
 return logits[:n],logits[n:],acts
def main():
 p=argparse.ArgumentParser();p.add_argument('--single-manifest',required=True,type=Path);p.add_argument('--six-manifest',required=True,type=Path);p.add_argument('--teacher',required=True,type=Path);p.add_argument('--imagenet-weights',required=True,type=Path);p.add_argument('--r0-students',required=True,type=Path,nargs=3);p.add_argument('--output',required=True,type=Path);p.add_argument('--batch-size',type=int,default=16);a=p.parse_args()
 teacher=load_resnet(a.teacher,100);pre=models.resnet18(weights=None);pre.load_state_dict(torch.load(a.imagenet_weights,map_location='cpu',weights_only=True));pre=pre.cuda().eval();students=[load_resnet(x,100) for x in a.r0_students]
 single=json.loads(a.single_manifest.read_text());single_arms=('isotropic158','vertical112','horizontal112','bbox_retarget112','attention_retarget112');six_arms=('uniform','bbox','background_hsmooth','object_decoded','background_decoded');six_mode={'uniform':'uniform','bbox':'compressed','background_hsmooth':'background_hsmooth','object_decoded':'object_decoded','background_decoded':'background_decoded'};scores={'single':{},'six':{}}
 for setting,arms in [('single',single_arms),('six',six_arms)]:
  for arm in arms:
   sums={k:0. for k in ('p1_teacher_kl20','p2_teacher_true_prob_drop','p3_teacher_layer2','p3_teacher_layer3','p4_imagenet_layer2','p4_imagenet_layer3','p5_r0_student_kl20','pixel_mae','negative_psnr','one_minus_ssim')};student_sums=[0.,0.,0.];count=0
   if setting=='single':
    entries=[]
    for row in single['records']:
     entries.append((row['source_path'],row['outputs'][arm]['decoded'],row['official_bbox224'][1:4:2],int(row['class'])))
   else:
    idx=PairedSourceIndex(a.six_manifest,six_mode[arm]);entries=[]
    for parent,row in enumerate(idx.rows):
     for source in (0,1):
      z=row['sources'][source];xmin,ymin,xmax,ymax=z['official_bbox_1indexed'];rh=z['raw_size'][1];entries.append((z['reference_path'],(idx,parent,source),((ymin-1)/rh*224,ymax/rh*224),row['class_id']))
   for off in range(0,len(entries),a.batch_size):
    batch=entries[off:off+a.batch_size];orig=[];var=[];boxes=[];labels=[]
    for op,vp,box,label in batch:
     with Image.open(op) as h:orig.append(tensor(h.convert('RGB')))
     if isinstance(vp,tuple):var.append(tensor(vp[0].load(vp[1],vp[2])))
     else:
      with Image.open(vp) as h:var.append(tensor(h.convert('RGB')))
     boxes.append(box);labels.append(label)
    o=torch.stack(orig).cuda();v=torch.stack(var).cuda();lab=torch.tensor(labels,device='cuda');n=len(batch)
    mae=(o-v).abs().mean((1,2,3));mse=(o-v).square().mean((1,2,3)).clamp_min(1e-12);sums['pixel_mae']+=float(mae.sum());sums['negative_psnr']+=float((10*torch.log10(mse)).sum());sums['one_minus_ssim']+=float(ssim_loss(o,v).sum())
    for flip in (False,True):
     oo=torch.flip(o,(-1,)) if flip else o;vv=torch.flip(v,(-1,)) if flip else v
     to,tv,ta=model_pair(teacher,oo,vv,TMEAN,TSTD,True);sums['p1_teacher_kl20']+=float(kl20(to,tv).sum())/2;sums['p2_teacher_true_prob_drop']+=float((F.softmax(to,1)[torch.arange(n,device='cuda'),lab]-F.softmax(tv,1)[torch.arange(n,device='cuda'),lab]).sum())/2
     for layer in ('layer2','layer3'):sums['p3_teacher_'+layer]+=sum(feature_distance(ta[layer][0],ta[layer][1],boxes))/2
     _,_,pa=model_pair(pre,oo,vv,IMEAN,ISTD,True)
     for layer in ('layer2','layer3'):sums['p4_imagenet_'+layer]+=sum(feature_distance(pa[layer][0],pa[layer][1],boxes))/2
     for si,student in enumerate(students):so,sv,_=model_pair(student,oo,vv,TMEAN,TSTD,False);value=float(kl20(so,sv).sum())/2;student_sums[si]+=value;sums['p5_r0_student_kl20']+=value/(2*len(students))*2
    count+=n
   row={k:v/count for k,v in sums.items()};row['p3_teacher_mean']=(row['p3_teacher_layer2']+row['p3_teacher_layer3'])/2;row['p4_imagenet_mean']=(row['p4_imagenet_layer2']+row['p4_imagenet_layer3'])/2;row['p5_student_individual']=[x/count for x in student_sums];scores[setting][arm]=row;print(setting,arm,row,flush=True)
 proxies=[k for k in next(iter(scores['single'].values())) if k!='p5_student_individual']+['p5_student42','p5_student43','p5_student44'];evaluation={}
 for proxy in proxies:
  evaluation[proxy]={};passed=True
  for setting,arms in [('single',single_arms),('six',six_arms)]:
   vals=([scores[setting][x]['p5_student_individual'][int(proxy[-2:])-42] for x in arms] if proxy.startswith('p5_student') else [scores[setting][x][proxy] for x in arms]);loss=[LOSSES[setting][x] for x in arms];rho=spearman(vals,loss);evaluation[proxy][setting]={'spearman':rho,'arms':list(arms),'scores':vals,'student_losses':loss};passed&=rho>=.9-1e-12
  s=scores['single'];z=scores['six'];get=lambda d,k:(d[k]['p5_student_individual'][int(proxy[-2:])-42] if proxy.startswith('p5_student') else d[k][proxy]);single_order=get(s,'bbox_retarget112')<get(s,'attention_retarget112') and get(s,'horizontal112')==max(get(s,x) for x in single_arms);six_order=get(z,'background_decoded')<get(z,'object_decoded');sixvals=[get(z,x) for x in six_arms];adj=abs(sorted(sixvals).index(get(z,'background_hsmooth'))-sorted(sixvals).index(get(z,'bbox')))<=1;close=abs(get(z,'background_hsmooth')-get(z,'bbox'))<=.25*(max(sixvals)-min(sixvals));evaluation[proxy]['critical_pairs']={'single_bbox_lt_attention_and_horizontal_worst':single_order,'six_background_lt_object':six_order,'six_smooth_approximately_bbox':adj and close,'smooth_rule':'adjacent rank and absolute gap <=25% of five-arm score range'};evaluation[proxy]['passes']=bool(passed and single_order and six_order and adj and close)
 result={'status':'complete','score_direction':'all scores transformed so larger means more distortion/worse','student_losses':LOSSES,'scores':scores,'evaluation':evaluation,'passing_proxies':[k for k,v in evaluation.items() if v['passes']]};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passing_proxies':result['passing_proxies'],'evaluation':evaluation},indent=2))
if __name__=='__main__':main()
