"""Add matched uniform 224x112 strips to an existing six-source manifest."""
import argparse,json,os
from pathlib import Path
from PIL import Image
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,type=Path);p.add_argument('--output-root',required=True,type=Path);a=p.parse_args();x=json.loads(a.manifest.read_text())
 for row in x['records']:
  packed=Image.new('RGB',(224,224))
  for sub,source in enumerate(row['sources']):
   with Image.open(source['reference_path']) as h:image=h.convert('RGB')
   packed.paste(image.resize((224,112),Image.Resampling.LANCZOS),(0,sub*112))
  out=a.output_root/row['class']/f"parent_{row['slot']}.png";out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.png.tmp');packed.save(tmp,format='PNG',compress_level=0);os.replace(tmp,out);row['uniform_packed_path']=str(out.resolve())
 x['uniform_encoding']='PIL Lanczos 224x224 to 224x112 uint8 PNG strip; two strips stacked; selected strip Lanczos decoded to 224x224 before augmentation'
 tmp=a.manifest.with_suffix('.json.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');os.replace(tmp,a.manifest);print(json.dumps({'status':'complete','parents':len(x['records'])}))
if __name__=='__main__':main()
