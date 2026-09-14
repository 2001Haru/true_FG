"""Create matched native-detail and low-resolution Aircraft bbox views."""
import argparse,hashlib,json,math,os,statistics
from pathlib import Path
from PIL import Image,ImageChops,ImageStat

FILL=(round(0.4865*255),round(0.5177*255),round(0.5425*255))
def sha(path):
 d=hashlib.sha256()
 with path.open('rb') as h:
  for b in iter(lambda:h.read(1048576),b''):d.update(b)
 return d.hexdigest()
def letterbox(image,size=224,layout=None):
 if layout is None:
  scale=min(size/image.width,size/image.height);new=(max(1,round(image.width*scale)),max(1,round(image.height*scale)));offset=((size-new[0])//2,(size-new[1])//2)
 else:new,offset=layout
 resized=image.resize(new,Image.Resampling.BILINEAR);canvas=Image.new('RGB',(size,size),FILL);canvas.paste(resized,offset);return canvas,new,offset
def stats(v):return {'mean':statistics.mean(v),'median':statistics.median(v),'min':min(v),'max':max(v)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--raw-images',required=True,type=Path);p.add_argument('--boxes',required=True,type=Path);p.add_argument('--r0-root',required=True,type=Path);p.add_argument('--output-root',required=True,type=Path);p.add_argument('--margin',type=float,default=.10);a=p.parse_args();boxes={}
 for line in a.boxes.read_text().splitlines():
  z=line.split();boxes[z[0]]=tuple(map(int,z[1:5]))
 records=[];areas=[];content_native=[];content_low=[];maes=[];pair_maes=[];cropped=0
 for class_dir in sorted(p for p in a.r0_root.iterdir() if p.is_dir()):
  for current in sorted(class_dir.iterdir()):
   image_id=current.stem;raw_path=a.raw_images/f'{image_id}.jpg'
   if not raw_path.is_file() or image_id not in boxes:raise RuntimeError(f'missing raw/box {image_id}')
   with Image.open(raw_path) as h:raw=h.convert('RGB')
   with Image.open(current) as h:low=h.convert('RGB')
   if low.size!=(224,224):raise RuntimeError(f'R0 is not 224: {current}')
   xmin,ymin,xmax,ymax=boxes[image_id];x0=xmin-1.;y0=ymin-1.;x1=float(xmax);y1=float(ymax);bw=x1-x0;bh=y1-y0
   x0=max(0.,x0-a.margin*bw);x1=min(float(raw.width),x1+a.margin*bw);y0=max(0.,y0-a.margin*bh);y1=min(float(raw.height),y1+a.margin*bh)
   native_box=(math.floor(x0),math.floor(y0),math.ceil(x1),math.ceil(y1));normalized=(native_box[0]/raw.width,native_box[1]/raw.height,native_box[2]/raw.width,native_box[3]/raw.height)
   low_box=(math.floor(normalized[0]*224),math.floor(normalized[1]*224),math.ceil(normalized[2]*224),math.ceil(normalized[3]*224));low_box=tuple(max(0,min(224,x)) for x in low_box)
   native_crop=raw.crop(native_box);low_crop=low.crop(low_box);native_out,nsize,noffset=letterbox(native_crop);low_out,lsize,loffset=letterbox(low_crop,layout=(nsize,noffset))
   outputs={"native_subject":native_out,"lowres_subject":low_out}
   for method,image in outputs.items():
    target=a.output_root/'selected'/method/'ipc3'/class_dir.name/current.name;target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_name(target.name+'.tmp');image.save(temporary,format='JPEG',quality=95,optimize=False);os.replace(temporary,target)
   full=native_box==(0,0,raw.width,raw.height);cropped+=not full;area=(native_box[2]-native_box[0])*(native_box[3]-native_box[1])/(raw.width*raw.height);areas.append(area);content_native.append(nsize[0]*nsize[1]/224**2);content_low.append(lsize[0]*lsize[1]/224**2)
   recomputed=raw.resize((224,224),Image.Resampling.BILINEAR);difference=ImageChops.difference(recomputed,low);mae=sum(ImageStat.Stat(difference).mean)/3;maes.append(mae);pair_mae=sum(ImageStat.Stat(ImageChops.difference(native_out,low_out)).mean)/3;pair_maes.append(pair_mae)
   records.append({'class':class_dir.name,'image_id':image_id,'raw_path':str(raw_path.resolve()),'r0_path':str(current.resolve()),'raw_size':list(raw.size),'official_bbox_1indexed':[xmin,ymin,xmax,ymax],'margin_fraction_each_side':a.margin,'native_crop_box_0indexed_exclusive':list(native_box),'normalized_window':list(normalized),'lowres_crop_box_224':list(low_box),'window_area_fraction':area,'full_frame':full,'native_letterbox_content_size':list(nsize),'native_letterbox_offset':list(noffset),'lowres_letterbox_content_size':list(lsize),'lowres_letterbox_offset':list(loffset),'r0_vs_raw_direct_resize_pixel_mae':mae,'native_vs_lowres_output_pixel_mae':pair_mae})
 if len(records)!=300:raise RuntimeError(f'expected 300 records, got {len(records)}')
 gate=cropped/len(records)>=.5
 manifest={'status':'complete','dataset':'A_imsize224','ipc':3,'source':'exact R0 seed0 300 image identities','raw_images':str(a.raw_images.resolve()),'boxes':str(a.boxes.resolve()),'bbox_format':'official xmin ymin xmax ymax, top-left=(1,1)','r0_preprocessing_audit':'decoded R0 compared with raw RGB direct bilinear warp224; no bbox used','margin_fraction_each_bbox_side':a.margin,'window_rule':'official bbox plus 10% width/height each side, clipped to raw image','output_rule':'native crop aspect-preserving bilinear fit in 224x224; Aircraft-mean RGB letterbox; JPEG quality95 optimize=False','lowres_rule':'same normalized window applied to existing R0 224x224 pixels, then forced into the exact native content size/offset; geometry matched, native detail unrecoverable','images':300,'cropped_images':cropped,'cropped_fraction':cropped/len(records),'training_gate':{'minimum_cropped_fraction':.5,'passed':gate},'window_area_fraction':stats(areas),'native_letterbox_content_fraction':stats(content_native),'lowres_letterbox_content_fraction':stats(content_low),'r0_vs_raw_direct_resize_pixel_mae':stats(maes),'native_vs_lowres_output_pixel_mae':stats(pair_maes),'records':records}
 a.output_root.mkdir(parents=True,exist_ok=True);out=a.output_root/'construction_manifest.json';tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');os.replace(tmp,out);print(json.dumps({k:manifest[k] for k in ('status','images','cropped_images','cropped_fraction','training_gate','window_area_fraction','native_letterbox_content_fraction','lowres_letterbox_content_fraction','r0_vs_raw_direct_resize_pixel_mae')},indent=2));
 if not gate:raise SystemExit(3)
if __name__=='__main__':main()
