"""Pinned data acquisition. Dry run by default; networking requires --download."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import stat
import sys
import urllib.parse
import urllib.request
import uuid
import zipfile

REPOSITORY = Path(__file__).resolve().parents[1]
REGISTRY = REPOSITORY / "data" / "registry.json"
CHUNK = 1024 * 1024


def safe_target(root, relative):
    name = PurePosixPath(str(relative).replace("\\", "/"))
    if name.is_absolute() or ".." in name.parts or ":" in str(name) or not name.parts:
        raise ValueError("Unsafe relative path: " + str(relative))
    root = Path(root).resolve()
    target = (root / str(name)).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError("Path escapes dataset directory: " + str(relative))
    return target


def destination(root, entry, dataset):
    prefix = "data/raw/" + dataset + "/"
    name = entry["local_path"]
    if not name.startswith(prefix):
        raise ValueError("Registry path does not match dataset: " + name)
    return safe_target(root, name[len(prefix):])


def check_file(path, entry):
    if not path.is_file():
        raise ValueError("Missing file: " + str(path))
    if path.stat().st_size != entry["bytes"]:
        raise ValueError("Byte count mismatch; existing file preserved: " + str(path))
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            h.update(chunk)
    if h.hexdigest() != entry["sha256"]:
        raise ValueError("SHA-256 mismatch; existing file preserved: " + str(path))


def new_part(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(path.name + ".part." + uuid.uuid4().hex)


def publish_part(part, target):
    if part.resolve().parent != target.resolve().parent:
        raise ValueError("Temporary and final file must share a directory")
    # Windows rename refuses an existing destination. POSIX rename would replace
    # it, so use exclusive hard-link publication there instead.
    if os.name == "nt":
        os.rename(part, target)
    else:
        os.link(part, target)
        part.unlink()


def require_https(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Only credential-free HTTPS URLs are accepted")


class HTTPSRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        require_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_file(entry, target):
    if target.exists():
        check_file(target, entry)
        return "existing_verified"
    require_https(entry["url"])
    part = new_part(target)
    total = 0
    h = hashlib.sha256()
    try:
        opener = urllib.request.build_opener(HTTPSRedirects())
        request = urllib.request.Request(entry["url"], headers={"User-Agent": "pmsm-data-acquisition/1.0"})
        with opener.open(request, timeout=30) as response:
            require_https(response.geturl())
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > entry["bytes"]:
                raise ValueError("Response exceeds pinned byte bound")
            with part.open("xb") as output:
                while True:
                    chunk = response.read(min(CHUNK, entry["bytes"] - total + 1))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > entry["bytes"]:
                        raise ValueError("Download exceeds pinned byte bound")
                    output.write(chunk)
                    h.update(chunk)
        if total != entry["bytes"] or h.hexdigest() != entry["sha256"]:
            raise ValueError("Downloaded bytes do not match pinned size/SHA-256")
        publish_part(part, target)
        return "downloaded_verified"
    except Exception as exc:
        suffix = "; partial file retained: " + str(part) if part.exists() else ""
        raise ValueError(str(exc) + suffix) from exc


def verify_zip(archive, entry, root, dataset, extract=False):
    check_file(archive, entry)
    required = {m["archive_path"]: m for m in entry["members"]}
    if len(required) != len(entry["members"]):
        raise ValueError("Duplicate registry ZIP member")
    with zipfile.ZipFile(archive) as z:
        files = {}
        seen = set()
        for info in z.infolist():
            safe_target(root, info.filename)
            name = info.filename.replace("\\", "/")
            folded = name.casefold()
            if folded in seen or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Duplicate or symbolic-link ZIP entry")
            seen.add(folded)
            if not info.is_dir():
                files[name] = info
        if set(files) != set(required):
            raise ValueError("ZIP members differ from the pinned dataset")
        for name, member in required.items():
            if files[name].file_size != member["bytes"]:
                raise ValueError("ZIP member byte count mismatch: " + name)
            target = destination(root, member, dataset)
            if target.exists():
                check_file(target, member)
        for name, member in required.items():
            target = destination(root, member, dataset)
            part = new_part(target) if extract and not target.exists() else None
            output = part.open("xb") if part else None
            total = 0
            h = hashlib.sha256()
            try:
                with z.open(files[name]) as stream:
                    while True:
                        chunk = stream.read(min(CHUNK, member["bytes"] - total + 1))
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > member["bytes"]:
                            raise ValueError("Expanded member exceeds pinned bound")
                        h.update(chunk)
                        if output:
                            output.write(chunk)
                if total != member["bytes"] or h.hexdigest() != member["sha256"]:
                    raise ValueError("Expanded member SHA-256 mismatch: " + name)
            finally:
                if output:
                    output.close()
            if part:
                publish_part(part, target)


def process_dataset(name, specification, root, mode):
    results = []
    for entry in specification["downloads"]:
        target = destination(root, entry, name)
        if mode == "dry_run":
            results.append({"url": entry["url"], "bytes": entry["bytes"], "sha256": entry["sha256"], "destination": str(target)})
            continue
        members = entry.get("members", [])
        # Validate every existing expected destination before any network call.
        if target.exists():
            check_file(target, entry)
        for member in members:
            member_target = destination(root, member, name)
            if member_target.exists():
                check_file(member_target, member)
        if members and not target.exists() and all(destination(root, m, name).is_file() for m in members):
            results.append({"file": str(target), "status": "extracted_members_verified_without_transport"})
            continue
        if mode == "download":
            status = download_file(entry, target)
        else:
            check_file(target, entry)
            status = "verified"
        if members:
            verify_zip(target, entry, root, name, extract=(mode == "download"))
        results.append({"file": str(target), "status": status, "verified_members": len(members)})
    return {"dataset": name, "mode": mode, "files": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "ipmsm", "paderborn", "rwth"], default="all")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--download", action="store_true", help="Explicitly allow HTTPS downloads and extraction")
    action.add_argument("--verify-only", action="store_true", help="Offline verification; no extraction or changes")
    parser.add_argument("--directory", type=Path, help="Existing/output root for one selected dataset; not allowed with all")
    args = parser.parse_args(argv)
    if args.directory is not None and args.dataset == "all":
        parser.error("--directory requires a single --dataset")
    mode = "download" if args.download else "verify_only" if args.verify_only else "dry_run"
    try:
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        names = list(registry["datasets"]) if args.dataset == "all" else [args.dataset]
        result = []
        for name in names:
            root = args.directory if args.directory is not None else REPOSITORY / "data" / "raw" / name
            result.append(process_dataset(name, registry["datasets"][name], root, mode))
        print(json.dumps({"status": "DRY_RUN" if mode == "dry_run" else "PASS", "mode": mode, "datasets": result}, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "mode": mode, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
