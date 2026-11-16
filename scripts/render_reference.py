"""Re-render the included saved manuscript snapshot; never run an experiment."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
from pmsm_repro.reporting import generate_from_directory


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Render saved manuscript reference tables/figures, without training, inference or timing.")
    parser.add_argument("--output", type=Path, default=REPOSITORY / "work" / "reference",
                        help="New output directory (default: repository work/reference); must not already exist.")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Reference output already exists; choose a new --output directory.")
    reference = REPOSITORY / "reference"
    provenance = json.loads((reference / "provenance.json").read_text(encoding="utf-8"))
    required = {"accuracy.csv", "profiles.csv", "timing.csv", "metadata.json"}
    entries = provenance["included_inputs"]
    if len(entries) != len(required) or {item["path"] for item in entries} != required:
        raise ValueError("Reference provenance must identify exactly the four explicit input files.")
    for item in entries:
        path = reference / item["path"]
        payload = path.read_bytes()
        if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise ValueError("Frozen reference checksum mismatch: " + item["path"])
    metadata = json.loads((reference / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("reference_snapshot") is not True or metadata.get("new_experiment") is not False:
        raise ValueError("Reference metadata must explicitly identify a saved snapshot.")
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    report = generate_from_directory(reference, output, mode="reference_snapshot", strict=True)
    print(json.dumps({"mode": report["mode"], "status": report["status"],
                      "output": str(output), "new_experiment": False,
                      "scope": "Saved manuscript reference only; no new training, inference or timing."}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
