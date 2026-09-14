"""Reuse frozen Aircraft bbox windows and warp native crops directly to 224."""
import argparse,json,os,statistics
from pathlib import Path
from PIL import Image
def stats(v):return {'mean':statistics.mean(v),'median':statistics.median(v),'min':min(v),'max':max(v)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--source-manifest',required=True,type=Path);p.add_argument('--output-root',required=True,type=Path);a=p.parse_args();source=json.loads(a.source_manifest.read_text());records=[];metrics={k:[] for k in ('r0_bbox_width','r0_bbox_height','r0_bbox_area_fraction','warp_bbox_width','warp_bbox_height','warp_bbox_area_fraction','warp_over_r0_width','warp_over_r0_height','warp_over_r0_area')}
 for row in source['records']:
  raw_path=Path(row['raw_path']);crop=tuple(row['native_crop_box_0indexed_exclusive']);W,H=row['raw_size'];xmin,ymin,xmax,ymax=row['official_bbox_1indexed'];bw=xmax-(xmin-1);bh=ymax-(ymin-1);cw=crop[2]-crop[0];ch=crop[3]-crop[1]
  with Image.open(raw_path) as h:image=h.convert('RGB').crop(crop).resize((224,224),Image.Resampling.BILINEAR)
  target=a.output_root/'selected/native_subject_warp/ipc3'/row['class']/f"{row['image_id']}.jpg";target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name(target.name+'.tmp');image.save(tmp,format='JPEG',quality=95,optimize=False);os.replace(tmp,target)
  rw=bw/W*224;rh=bh/H*224;ww=bw/cw*224;wh=bh/ch*224
  values={'r0_bbox_width':rw,'r0_bbox_height':rh,'r0_bbox_area_fraction':rw*rh/224**2,'warp_bbox_width':ww,'warp_bbox_height':wh,'warp_bbox_area_fraction':ww*wh/224**2,'warp_over_r0_width':ww/rw,'warp_over_r0_height':wh/rh,'warp_over_r0_area':(ww*wh)/(rw*rh)}
  for k,v in values.items():metrics[k].append(v)
  records.append({'class':row['class'],'image_id':row['image_id'],'raw_path':str(raw_path.resolve()),'frozen_crop_box':list(crop),'normalized_window':row['normalized_window'],'output_path':str(target.resolve()),**values})
 if len(records)!=300:raise RuntimeError(f'expected 300 images, got {len(records)}')
 result={'status':'complete','dataset':'A_imsize224','ipc':3,'source_manifest':str(a.source_manifest.resolve()),'window_identity':'exact frozen bbox+10% native crop box from source manifest','output':'native RGB crop direct PIL bilinear warp224; JPEG quality95 optimize=False','images':300,'metrics':{k:stats(v) for k,v in metrics.items()},'records':records};a.output_root.mkdir(parents=True,exist_ok=True);out=a.output_root/'construction_manifest.json';tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');os.replace(tmp,out);print(json.dumps({'status':'complete','images':300,'metrics':result['metrics']},indent=2))
if __name__=='__main__':main()
