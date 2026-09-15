"""Materialize an audited selection manifest as a collision-safe ImageFolder symlink tree."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--audit-output", required=True, type=Path)
    a = p.parse_args();x=json.loads(a.manifest.read_text())
    assert x["status"] == "complete" and len(x["images"]) == x["classes"] * x["ipc"]
    records=[]
    for row in sorted(x["images"], key=lambda r:(r["class_folder"],r["slot"],r["source_path"])):
        source=Path(row["source_path"]).resolve();assert source.is_file()
        destination=a.output_root/row["class_folder"]/f"slot{int(row['slot']):02d}_{source.name}"
        destination.parent.mkdir(parents=True,exist_ok=True)
        if os.path.lexists(destination):
            if destination.resolve()!=source:raise RuntimeError(f"collision: {destination}")
        else:destination.symlink_to(source)
        records.append({"class_folder":row["class_folder"],"slot":int(row["slot"]),
                        "source_path":str(source),"materialized_path":str(destination.resolve()),
                        "source_sha256":sha256(source)})
    files=[p for p in a.output_root.rglob('*') if p.is_file()]
    assert len(files)==len(records)
    digest=hashlib.sha256()
    for row in records:
        digest.update(row["class_folder"].encode());digest.update(str(row["slot"]).encode())
        digest.update(row["source_path"].encode());digest.update(bytes.fromhex(row["source_sha256"]))
    result={"status":"complete","selection_manifest":str(a.manifest.resolve()),
            "selection_manifest_sha256":sha256(a.manifest),"output_root":str(a.output_root.resolve()),
            "classes":x["classes"],"ipc":x["ipc"],"images":len(records),
            "identity_sha256":digest.hexdigest(),"records":records}
    a.audit_output.parent.mkdir(parents=True,exist_ok=True);tmp=a.audit_output.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result,indent=2)+'\n');os.replace(tmp,a.audit_output)
    print(json.dumps({k:result[k] for k in ('status','classes','ipc','images','identity_sha256')}))


if __name__=='__main__':main()
