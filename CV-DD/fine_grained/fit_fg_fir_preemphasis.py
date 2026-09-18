"""Fit Aircraft F1-v1 capacity on a dataset-specific public600 and fixed axis."""
import argparse,json,os
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from prepare_aircraft_retarget_compression import encode_vertical_area
from paired_source_dataset import decode_vertical

BANDS=((0.,.25,7),(.25,.65,7),(.65,1.0000001,5))
def kernel(raw,length):
 side=.30*torch.tanh(raw);return torch.cat((side,(1-2*side.sum())[None],side.flip(0)))
def density(box,axis,target,b):
 import math
 lo,hi=(box[1],box[3]) if axis=='row' else (box[0],box[2]);first=max(0,min(223,math.floor(lo)));last=max(first+1,min(224,math.ceil(hi)))
 mask=np.zeros(224,dtype=bool);mask[first:last]=True;h=int(mask.sum());bg=224-h;sd=min(.9,(target-b*bg)/h);bd=(target-sd*h)/bg if bg else target/224
 # The area encoder integrates this vector and requires an integer terminal
 # coordinate.  Keep it in float64: float32 accumulation misses 75 by about
 # 1e-4 on CUB/Cars and used to abort F1 fitting before any Student run.
 x=np.where(mask,sd,bd).astype(np.float64)
 x[-1]+=float(target)-float(x.sum())
 if x.min()<=0 or abs(float(x.sum())-target)>1e-9:raise RuntimeError((x.min(),x.sum(),target))
 return x
def apply_axis(x,rho,raws,axis):
 if axis=='column':x=x.transpose(-1,-2)
 out=torch.zeros_like(x);covered=torch.zeros_like(rho,dtype=torch.bool)
 for raw,(lo,hi,length) in zip(raws,BANDS):
  k=kernel(raw,length);r=length//2;p=F.pad(x,(0,0,r,r),mode='reflect');filtered=F.conv2d(p,k.view(1,1,length,1).repeat(3,1,1,1),groups=3)
  mask=((rho>=lo)&(rho<hi))[:,None,:,None];out=torch.where(mask,filtered,out);covered|=mask[:,0,:,0]
 if not covered.all():raise RuntimeError('uncovered')
 out=out.clamp(0,1);return out.transpose(-1,-2) if axis=='column' else out
def moments(conv,x,mean,std):
 z=conv((x-mean)/std);return z.mean((0,2,3)),torch.log(z.var((0,2,3),unbiased=False)+1e-6)
def main():
 p=argparse.ArgumentParser();p.add_argument('--public-manifest',required=True,type=Path);p.add_argument('--bbox-manifest',required=True,type=Path)
 p.add_argument('--base-manifest',required=True,type=Path);p.add_argument('--imagenet-weights',required=True,type=Path);p.add_argument('--output-manifest',required=True,type=Path)
 p.add_argument('--audit',required=True,type=Path);p.add_argument('--steps',type=int,default=400);p.add_argument('--batch-size',type=int,default=20);a=p.parse_args()
 public=json.load(open(a.public_manifest));bbox=json.load(open(a.bbox_manifest));boxes={x['prepared_path']:x for x in bbox['rows']};base=json.load(open(a.base_manifest))
 axis=base['compression_axis'];b=float(base['background_minimum']);heights=base['strip_heights'];clear=[];decoded=[];densities=[]
 for index,value in enumerate(public['selected']):
  path=Path(value).resolve();meta=boxes[str(path)];image=np.asarray(Image.open(path).convert('RGB'),np.uint8);rho=density(meta['box224_xyxy'],axis,int(heights[index%3]),b)
  oriented=image if axis=='row' else image.transpose(1,0,2);encoded,_=encode_vertical_area(oriented,rho);recon=np.asarray(decode_vertical(encoded,rho),np.uint8)
  if axis=='column':recon=recon.transpose(1,0,2)
  clear.append(torch.from_numpy(image.copy()).permute(2,0,1).float()/255);decoded.append(torch.from_numpy(recon.copy()).permute(2,0,1).float()/255);densities.append(torch.from_numpy(rho.copy()))
 clear,decoded,densities=torch.stack(clear),torch.stack(decoded),torch.stack(densities)
 model=models.resnet18(weights=None);model.load_state_dict(torch.load(a.imagenet_weights,map_location='cpu',weights_only=True));conv=model.conv1.cuda().eval().requires_grad_(False)
 mean=torch.tensor(base['normalization_mean'],device='cuda')[None,:,None,None];std=torch.tensor(base['normalization_std'],device='cuda')[None,:,None,None]
 raws=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros((n-1)//2,device='cuda')) for _,_,n in BANDS]);opt=torch.optim.Adam(raws.parameters(),lr=.03,betas=(.9,.999));gen=torch.Generator().manual_seed(20260918);history=[]
 def loss_for(ids):
  x=clear[ids].cuda();d=decoded[ids].cuda();rho=densities[ids].cuda();tm,tv=moments(conv,x,mean,std);am,av=moments(conv,apply_axis(d,rho,raws,axis),mean,std);lm=((am-tm).square()/(tv.exp()+1e-6)).mean();lv=(av-tv).square().mean();return lm+lv,lm,lv
 with torch.no_grad():initial=tuple(float(v) for v in loss_for(torch.arange(600)))
 for step in range(a.steps):
  ids=torch.randint(600,(a.batch_size,),generator=gen);loss,lm,lv=loss_for(ids);opt.zero_grad();loss.backward();opt.step()
  if step in (0,9,49,99,199,a.steps-1):history.append({'step':step+1,'loss':float(loss),'mean':float(lm),'logvar':float(lv)})
 with torch.no_grad():final=tuple(float(v) for v in loss_for(torch.arange(600)))
 bands=[]
 for raw,(lo,hi,length) in zip(raws,BANDS):
  values=kernel(raw,length).detach().cpu().double().tolist();bands.append({'density_min':lo,'density_max':hi,'include_upper':hi>1,'length':length,'kernel':values,'dc_gain':sum(values),'symmetric_max_error':max(abs(x-y) for x,y in zip(values,values[::-1]))})
 config={'protocol':'fg_density_binned_fir_preemphasis_v1','dataset':base['dataset'],'axis':axis,'fit_images':600,'fit_source':'dataset-specific public non-storage train images',
  'encoded_heights':heights,'objective':'ImageNet-V1 ResNet18 conv1 channel mean standardized MSE + log-variance MSE','steps':a.steps,'batch_size':a.batch_size,
  'coefficient_bound':.30,'bands':bands};base['fir_preemphasis']=config;base['protocol']+='+fir_preemphasis_v1';a.output_manifest.parent.mkdir(parents=True,exist_ok=True)
 tmp=a.output_manifest.with_suffix('.tmp');tmp.write_text(json.dumps(base,indent=2)+'\n');os.replace(tmp,a.output_manifest)
 audit={'status':'complete','configuration':config,'initial_loss':{'total':initial[0],'mean':initial[1],'logvar':initial[2]},'final_loss':{'total':final[0],'mean':final[1],'logvar':final[2]},
  'relative_reduction':1-final[0]/initial[0],'history':history,'public_manifest':str(a.public_manifest),'selected_exclusion_overlap':public['selected_exclusion_overlap']}
 a.audit.parent.mkdir(parents=True,exist_ok=True);a.audit.write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
