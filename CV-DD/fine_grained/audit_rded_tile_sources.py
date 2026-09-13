"""Audit IPC3 RDED stored mosaics and their underlying tile source identities."""
import argparse,json,os,statistics
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--generation-manifest',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();x=json.loads(a.generation_manifest.read_text())
 if x.get('status')!='complete':raise RuntimeError('generation manifest incomplete')
 rows=x['outputs']['3'];by={i:[] for i in range(100)}
 for row in rows:
  if len(row['quadrants'])!=4:raise RuntimeError('mosaic does not have four tiles')
  by[int(row['class_id'])].extend(q['source_sha256'] for q in row['quadrants'])
 counts=[len(set(by[i])) for i in range(100)]
 payload={'status':'complete','generation_manifest':str(a.generation_manifest.resolve()),'stored_mosaics':len(rows),'classes':100,'ipc':3,'tile_positions_per_class':12,
  'unique_source_images_total':sum(counts),'unique_sources_per_class':counts,'unique_sources_per_class_min':min(counts),'unique_sources_per_class_mean':statistics.mean(counts),'unique_sources_per_class_max':max(counts),'new_high_resolution_information':0}
 if len(rows)!=300 or min(counts)<=0:raise RuntimeError('invalid source audit')
 a.output.parent.mkdir(parents=True,exist_ok=True);t=a.output.with_suffix('.tmp');t.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');os.replace(t,a.output);print(json.dumps(payload,sort_keys=True))
if __name__=='__main__':main()
