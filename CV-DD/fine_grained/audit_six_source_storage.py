"""Quantify image and density-map storage for the six-source BBox packing experiment."""

import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    payload=json.loads(a.manifest.read_text());parents=payload['records'];sources=[s for r in parents for s in r['sources']]
    packed_paths={Path(r['packed_path']) for r in parents};reference_paths={Path(s['reference_path']) for s in sources}
    density_json=sum(len(json.dumps(s['density'],separators=(',',':')).encode()) for s in sources)
    vectors=len(sources);entries=sum(len(s['density']) for s in sources)
    intervals=[];unique_counts=[]
    for source in sources:
        d=source['density'];xmin,ymin,xmax,ymax=source['official_bbox_1indexed'];raw_h=source['raw_size'][1]
        import math
        start=max(0,min(223,math.floor((ymin-1.)/raw_h*224.)))
        end=max(start+1,min(224,math.ceil(ymax/raw_h*224.)))
        reconstructed=[];h=end-start;bg=224-h
        if bg==0: reconstructed=[.5]*224
        else:
            subject=min(.9,(112-.1*bg)/h);back=(112-subject*h)/bg
            reconstructed=[subject if start<=i<end else back for i in range(224)]
        if max(abs(x-y) for x,y in zip(d,reconstructed))>1e-10:raise RuntimeError('boundary representation is not exact')
        intervals.append((start,end));unique_counts.append(len({round(v,12) for v in d}))
    packed_bytes=sum(path.stat().st_size for path in packed_paths);reference_bytes=sum(path.stat().st_size for path in reference_paths)
    result={'status':'complete','parents':len(parents),'sources':vectors,'density_entries':entries,
            'packed_png_bytes':packed_bytes,'reference_jpeg_bytes':reference_bytes,'manifest_bytes':a.manifest.stat().st_size,
            'density_only_compact_json_bytes':density_json,'generic_float32_density_bytes':entries*4,
            'generic_float16_density_bytes':entries*2,'bbox_exact_uint8_boundaries_bytes':vectors*2,
            'bbox_exact_uint16_boundaries_bytes':vectors*4,
            'mapping_over_packed_percent':{
                'current_density_json':100*density_json/packed_bytes,'float32':100*entries*4/packed_bytes,
                'float16':100*entries*2/packed_bytes,'uint8_boundaries':100*vectors*2/packed_bytes},
            'full_manifest_over_packed_percent':100*a.manifest.stat().st_size/packed_bytes,
            'bbox_mapping_property':'each 224-row vector has one contiguous subject interval and two analytically derived density values; global rule constants plus uint8 start/end reconstruct it exactly',
            'unique_density_values_per_source':{'min':min(unique_counts),'max':max(unique_counts)},
            'container_pixel_budget':len(parents)*224*224,'uncompressed_source_pixel_count':vectors*224*224}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
