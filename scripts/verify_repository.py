"""Verify a release's file hashes without importing scientific dependencies."""
from pathlib import Path
import hashlib,json,sys

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/"provenance/release-manifest.json"

def main():
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries=manifest["files"]
    names=[row["path"] for row in entries]
    if len(names)!=len(set(names)):raise ValueError("Duplicate manifest path")
    errors=[]
    for row in entries:
        path=ROOT/row["path"]
        if not path.resolve().is_relative_to(ROOT) or path.is_symlink():
            errors.append("unsafe path: "+row["path"]);continue
        if not path.is_file():errors.append("missing: "+row["path"]);continue
        if path.stat().st_size!=row["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest()!=row["sha256"]:
            errors.append("changed: "+row["path"])
    result={"status":"FAIL" if errors else "PASS","checked_files":len(entries),
            "version":manifest["version"],"errors":errors,
            "scope":"Listed release files only; raw inputs and generated work products have separate manifests."}
    print(json.dumps(result,indent=2))
    return int(bool(errors))

if __name__=="__main__":raise SystemExit(main())
