"""Build bbox-warp test views, cross-evaluate final students, and render examples."""
import argparse,json,math,os,statistics
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image,ImageDraw,ImageFont
from torch.utils.data import DataLoader
from torchvision import datasets,models,transforms

MEAN=(.4865,.5177,.5425);STD=(.2124,.2051,.2375)
def stats(v):v=list(map(float,v));return {'mean':statistics.mean(v),'sample_std':statistics.stdev(v),'values':v}
def boxes(path):
 out={}
 for line in path.read_text().splitlines():z=line.split();out[z[0]]=tuple(map(int,z[1:5]))
 return out
def window(box,size,margin=.1):
 W,H=size;xmin,ymin,xmax,ymax=box;x0=xmin-1.;y0=ymin-1.;x1=float(xmax);y1=float(ymax);w=x1-x0;h=y1-y0
 return (max(0,math.floor(x0-margin*w)),max(0,math.floor(y0-margin*h)),min(W,math.ceil(x1+margin*w)),min(H,math.ceil(y1+margin*h)))
def build_test(args,b):
 records=[];areas=[]
 for d in sorted(p for p in args.standard_test.iterdir() if p.is_dir()):
  for current in sorted(d.iterdir()):
   raw=args.raw_images/current.name
   if not raw.is_file() or current.stem not in b:raise RuntimeError(f'missing {current.name}')
   with Image.open(raw) as h:image=h.convert('RGB');crop=window(b[current.stem],image.size);area=(crop[2]-crop[0])*(crop[3]-crop[1])/(image.width*image.height);output=image.crop(crop).resize((224,224),Image.Resampling.BILINEAR)
   target=args.bbox_test/d.name/current.name;target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name(target.name+'.tmp');output.save(tmp,format='JPEG',quality=95,optimize=False);os.replace(tmp,target);areas.append(area);records.append({'class':d.name,'image_id':current.stem,'raw':str(raw.resolve()),'standard':str(current.resolve()),'crop':list(crop),'window_area_fraction':area,'output':str(target.resolve())})
 if len(records)!=3333:raise RuntimeError(f'test count {len(records)}')
 result={'status':'complete','images':len(records),'classes':100,'window':'official bbox + 10% each side, exact train construction rule','output':'native crop direct PIL bilinear warp224 JPEG quality95','area':{'mean':statistics.mean(areas),'median':statistics.median(areas),'min':min(areas),'max':max(areas)},'records':records};args.test_manifest.parent.mkdir(parents=True,exist_ok=True);args.test_manifest.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');return result
def evaluate(checkpoint,root,device,workers):
 dataset=datasets.ImageFolder(root,transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)]));loader=DataLoader(dataset,batch_size=256,shuffle=False,num_workers=workers,pin_memory=True,persistent_workers=workers>0);model=models.resnet18(weights=None);model.fc=nn.Linear(model.fc.in_features,100);payload=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(payload['state_dict'],strict=True);model.eval().to(device);correct=total=0;nll=0.
 with torch.no_grad():
  for images,target in loader:
   images=images.to(device,non_blocking=True);target=target.to(device,non_blocking=True);logits=model(images);correct+=int(logits.argmax(1).eq(target).sum());nll+=float(F.cross_entropy(logits,target,reduction='sum'));total+=len(target)
 return {'images':total,'top1':100*correct/total,'nll':nll/total,'checkpoint_epoch':payload['epoch']}
def montage(args,b):
 source=json.loads(args.train_manifest.read_text());ordered=sorted(source['records'],key=lambda x:x['window_area_fraction']);indices=[round(i*(len(ordered)-1)/9) for i in range(10)];rows=[ordered[i] for i in indices];cell=224;label_h=28;header=34;canvas=Image.new('RGB',(cell*4,header+len(rows)*(cell+label_h)),(245,245,245));draw=ImageDraw.Draw(canvas);font=ImageFont.load_default();titles=('Raw + frozen window','R0 full warp224','BBox warp224','BBox letterbox224')
 for col,title in enumerate(titles):draw.text((col*cell+5,10),title,fill=(0,0,0),font=font)
 for row_id,row in enumerate(rows):
  y=header+row_id*(cell+label_h);raw=Image.open(row['raw_path']).convert('RGB');scale=min(cell/raw.width,cell/raw.height);size=(round(raw.width*scale),round(raw.height*scale));thumb=raw.resize(size,Image.Resampling.BILINEAR);panel=Image.new('RGB',(cell,cell),(220,220,220));offset=((cell-size[0])//2,(cell-size[1])//2);panel.paste(thumb,offset);d=ImageDraw.Draw(panel);crop=row['native_crop_box_0indexed_exclusive'];d.rectangle((offset[0]+crop[0]*scale,offset[1]+crop[1]*scale,offset[0]+crop[2]*scale,offset[1]+crop[3]*scale),outline=(255,0,0),width=2);canvas.paste(panel,(0,y))
  canvas.paste(Image.open(row['r0_path']).convert('RGB'),(cell,y));warp=args.warp_train/row['class']/f"{row['image_id']}.jpg";letter=args.letterbox_train/row['class']/f"{row['image_id']}.jpg";canvas.paste(Image.open(warp).convert('RGB'),(cell*2,y));canvas.paste(Image.open(letter).convert('RGB'),(cell*3,y));draw.text((5,y+cell+5),f"{row['image_id']} area={row['window_area_fraction']:.3f}",fill=(0,0,0),font=font)
 args.montage.parent.mkdir(parents=True,exist_ok=True);canvas.save(args.montage)
 return {'rows':len(rows),'sampling':'10 evenly spaced ranks of frozen train-window area','columns':list(titles),'path':str(args.montage.resolve())}
def main():
 p=argparse.ArgumentParser();p.add_argument('--raw-images',required=True,type=Path);p.add_argument('--boxes',required=True,type=Path);p.add_argument('--standard-test',required=True,type=Path);p.add_argument('--bbox-test',required=True,type=Path);p.add_argument('--test-manifest',required=True,type=Path);p.add_argument('--train-manifest',required=True,type=Path);p.add_argument('--warp-train',required=True,type=Path);p.add_argument('--letterbox-train',required=True,type=Path);p.add_argument('--montage',required=True,type=Path);p.add_argument('--r0-checkpoints',required=True,type=Path,nargs=3);p.add_argument('--bbox-checkpoints',required=True,type=Path,nargs=3);p.add_argument('--r0-results',required=True,type=Path,nargs=3);p.add_argument('--bbox-results',required=True,type=Path,nargs=3);p.add_argument('--output',required=True,type=Path);p.add_argument('--workers',type=int,default=8);p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(4);b=boxes(a.boxes);test=build_test(a,b) if not a.test_manifest.is_file() else json.loads(a.test_manifest.read_text());visual=montage(a,b);rows=[];errors=[]
 for model,checkpoints,results in (('r0',a.r0_checkpoints,a.r0_results),('bbox_warp',a.bbox_checkpoints,a.bbox_results)):
  for seed,checkpoint,result_path in zip((42,43,44),checkpoints,results):
   recorded=json.loads(result_path.read_text());standard=evaluate(checkpoint,a.standard_test,a.device,a.workers);bbox=evaluate(checkpoint,a.bbox_test,a.device,a.workers)
   if standard['checkpoint_epoch']!=400 or abs(standard['top1']-recorded['final_epoch_top1'])>1e-9:errors.append({'model':model,'seed':seed,'error':'standard re-evaluation mismatch','computed':standard['top1'],'recorded':recorded['final_epoch_top1']})
   rows.append({'model':model,'student_seed':seed,'standard':standard,'bbox_warp_test':bbox,'bbox_minus_standard_top1':bbox['top1']-standard['top1'],'bbox_minus_standard_nll':bbox['nll']-standard['nll']})
 by={m:{r['student_seed']:r for r in rows if r['model']==m} for m in ('r0','bbox_warp')};summary={}
 for model,z in by.items():summary[model]={'standard_top1':stats(z[s]['standard']['top1'] for s in z),'bbox_top1':stats(z[s]['bbox_warp_test']['top1'] for s in z),'bbox_minus_standard_top1':stats(z[s]['bbox_minus_standard_top1'] for s in z),'standard_nll':stats(z[s]['standard']['nll'] for s in z),'bbox_nll':stats(z[s]['bbox_warp_test']['nll'] for s in z)}
 gaps={view:stats(by['bbox_warp'][s][view]['top1']-by['r0'][s][view]['top1'] for s in (42,43,44)) for view in ('standard','bbox_warp_test')};interaction=stats((by['bbox_warp'][s]['bbox_minus_standard_top1']-by['r0'][s]['bbox_minus_standard_top1']) for s in (42,43,44));result={'status':'complete' if not errors else 'failed','diagnostic':'official test bbox used only for post-hoc cross evaluation; formal standard-test metrics unchanged','test_construction':{k:test[k] for k in ('images','classes','window','output','area')},'models':summary,'bbox_student_minus_r0_gap':gaps,'view_match_interaction':interaction,'rows':rows,'visualization':visual,'errors':errors};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':result['status'],'models':summary,'bbox_student_minus_r0_gap':gaps,'view_match_interaction':interaction,'visualization':visual,'errors':errors},indent=2));
 if result['status']!='complete':raise SystemExit(1)
if __name__=='__main__':main()
