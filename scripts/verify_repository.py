"""Offline file integrity and historical syntax checks; no scientific import."""
from pathlib import Path,PurePosixPath
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
    failures=[];checks=0
    def check(label,ok):
        nonlocal checks
        checks+=1
        if not ok:failures.append(label)
    manifest=json.loads((ROOT/'SHA256SUMS.json').read_text(encoding='utf-8'))
    seen=set()
    for item in manifest['files']:
        name=item['path'];parts=PurePosixPath(name)
        safe=not parts.is_absolute() and '..' not in parts.parts and ':' not in name
        path=ROOT/name
        safe=safe and path.resolve().is_relative_to(ROOT) and not path.is_symlink()
        check('safe '+name,safe)
        if not safe:continue
        check('unique '+name,name not in seen);seen.add(name)
        check('hash '+name,path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'])
    check('provenance input listed in manifest','provenance/file-lineage.json' in seen)
    if failures:
        print(json.dumps({'status':'FAIL','checks':checks,'failures':failures},indent=2))
        return 1
    lineage=json.loads((ROOT/'provenance/file-lineage.json').read_text(encoding='utf-8'))
    python_count=0
    for item in lineage['files']:
        name=item['path'];path=ROOT/name
        safe=not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts and ':' not in name
        safe=safe and path.resolve().is_relative_to(ROOT) and not path.is_symlink()
        check('safe lineage '+name,safe)
        if not safe:continue
        check('lineage input listed in manifest '+name,name in seen)
        if name not in seen:continue
        blob=path.read_bytes()
        check('published provenance '+name,hashlib.sha256(blob).hexdigest()==item['published_sha256'])
        if item['category']=='historical_source':
            python_count+=1
            check('original source '+name,item['original_sha256']==item['published_sha256'])
            try:ast.parse(blob.decode('utf-8-sig'))
            except (SyntaxError,UnicodeError):check('syntax '+name,False)
        if path.suffix=='.json':
            try:json.loads(blob)
            except (ValueError,UnicodeError):check('JSON '+name,False)
    check('40 original Python sources',python_count==40)
    print(json.dumps({'status':'PASS' if not failures else 'FAIL','checks':checks,
        'historical_python_files':python_count,'failures':failures,
        'scope':'Delivered-file hashes, provenance and static syntax only; no training, inference, timing or network.'},indent=2))
    return bool(failures)
if __name__=='__main__':raise SystemExit(main())
