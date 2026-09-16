"""Render paired uncompressed/BBox-strip/decode examples for six-source packing."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFont

from paired_source_dataset import decode_vertical


def font(size, bold=False):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    try: return ImageFont.truetype(str(path), size)
    except OSError: return ImageFont.load_default()


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--classes',default='000,014,032,048,068,094');p.add_argument('--slot',type=int,default=0);a=p.parse_args()
    manifest=json.loads(a.manifest.read_text());wanted=set(a.classes.split(','));rows=[r for r in manifest['records'] if r['class'] in wanted and r['slot']==a.slot]
    rows=sorted(rows,key=lambda r:r['class']);tile=224;gap=14;left=190;head=58;label=30
    headers=('Source A original','Source A decoded','Stored 2-strip PNG','Source B decoded','Source B original')
    canvas=Image.new('RGB',(left+5*tile+4*gap+16,head+len(rows)*(tile+label+gap)+8),'white');draw=ImageDraw.Draw(canvas)
    hf,rf,sf=font(19,True),font(16,True),font(13)
    for col,text in enumerate(headers):draw.text((left+col*(tile+gap),15),text,fill='black',font=hf)
    diffs=Image.new('RGB',(left+2*tile+gap+16,head+len(rows)*(tile+label+gap)+8),'white');dd=ImageDraw.Draw(diffs)
    dd.text((left,15),'Source A |difference| ×4',fill='black',font=hf);dd.text((left+tile+gap,15),'Source B |difference| ×4',fill='black',font=hf)
    for ridx,row in enumerate(rows):
        y=head+ridx*(tile+label+gap);sources=row['sources'];packed=Image.open(row['packed_path']).convert('RGB');images=[];delta=[]
        for sub in (0,1):
            original=Image.open(sources[sub]['reference_path']).convert('RGB');strip=packed.crop((0,sub*112,224,(sub+1)*112));decoded=decode_vertical(strip,sources[sub]['density'])
            images.append((original,decoded));delta.append(ImageEnhance.Brightness(ImageChops.difference(decoded,original)).enhance(4))
        shown=(images[0][0],images[0][1],packed,images[1][1],images[1][0])
        draw.text((8,y+70),f"class {row['class']}  parent {row['slot']}\nA {sources[0]['identity']}\nB {sources[1]['identity']}",fill='black',font=rf,spacing=6)
        dd.text((8,y+70),f"class {row['class']}  parent {row['slot']}\nA {sources[0]['identity']}\nB {sources[1]['identity']}",fill='black',font=rf,spacing=6)
        for col,image in enumerate(shown):
            x=left+col*(tile+gap);canvas.paste(image,(x,y));draw.rectangle((x,y,x+223,y+223),outline=(120,120,120));
        for col,image in enumerate(delta):
            x=left+col*(tile+gap);diffs.paste(image,(x,y));dd.rectangle((x,y,x+223,y+223),outline=(120,120,120))
        draw.text((left,y+tile+6),'uncompressed',fill='black',font=sf);draw.text((left+tile+gap,y+tile+6),'inverse-mapped',fill='black',font=sf)
        draw.text((left+2*(tile+gap),y+tile+6),'top=A, bottom=B',fill='black',font=sf)
    a.output_dir.mkdir(parents=True,exist_ok=True);contact=a.output_dir/'six_source_bbox_contact_sheet.png';difference=a.output_dir/'six_source_bbox_difference_x4.png'
    canvas.save(contact);diffs.save(difference);print(json.dumps({'contact':str(contact),'difference':str(difference),'rows':len(rows)},indent=2))
if __name__=='__main__':main()
