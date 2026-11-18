"""Paired change in residual-minus-matched-direct RMSE across two refitted arms."""
import argparse
import csv
import hashlib
import io
import json
import math
import statistics as stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "reference/v007/analysis/failure-profile-corrections.csv"
FROZEN = ROOT / "reference/v013/initialization-contrast"
ARMS = ("original", "own_boundary")
COHORTS = (("paderborn", "development"), ("paderborn", "confirmation"),
           ("ipmsm", "calibration"), ("ipmsm", "pilot"), ("ipmsm", "confirmation"))


def csv_text(records):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(records[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue()


def derive(source):
    roles = json.loads((ROOT / "configs/study.json").read_text(encoding="utf-8"))["datasets"]
    expected = {(ds, role, int(pid), arm) for ds, role in COHORTS
                for pid in roles[ds]["roles"][role] for arm in ARMS}
    ledger, scopes = {}, set()
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["dataset"], row["role"], int(row["profile"]), row["arm"])
            if key not in expected or key in ledger:
                raise ValueError(f"Unexpected or duplicate profile/arm: {key}")
            residual, direct = (float(row[k]) for k in ("linear_residual_rmse_c", "direct200_rmse_c"))
            if not all(math.isfinite(v) and v >= 0 for v in (residual, direct)):
                raise ValueError(f"Invalid RMSE at {key}")
            difference = residual - direct
            if not math.isclose(difference, float(row["residual_minus_matched_rmse_c"]),
                                rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"Within-arm difference mismatch at {key}")
            ledger[key] = (residual, direct, difference, int(row["n"]))
            scopes.add(row.get("smoke_only", "false").lower())
    if set(ledger) != expected or len(scopes) != 1 or not scopes <= {"true", "false"}:
        raise ValueError("Missing profile arms or inconsistent run scope")
    smoke = scopes == {"true"}
    profiles, cohorts = [], []
    for ds, role in COHORTS:
        pairs = []
        for pid in sorted(roles[ds]["roles"][role]):
            ambient, own = [ledger[(ds, role, pid, a)] for a in ARMS]
            if ambient[3] <= 0 or ambient[3] != own[3]:
                raise ValueError(f"Unmatched row counts: {ds}/{role}/{pid}")
            pairs.append((*ambient[:3], *own[:3], own[2] - ambient[2]))
            profiles.append(dict(dataset=ds, role=role, profile=pid, n=ambient[3],
                ambient_residual_minus_direct_c=ambient[2], channel_residual_minus_direct_c=own[2],
                change_in_contrast_c=pairs[-1][-1], smoke_only=smoke))
        ar, ad, ac, cr, cd, cc, changes = zip(*pairs)
        deletion = [stats.mean(changes[:i] + changes[i + 1:]) for i in range(len(changes))]
        cohorts.append(dict(dataset=ds, role=role, profiles=len(pairs),
            ambient_residual_macro_rmse_c=stats.mean(ar), ambient_direct_macro_rmse_c=stats.mean(ad),
            ambient_contrast_c=stats.mean(ac), channel_residual_macro_rmse_c=stats.mean(cr),
            channel_direct_macro_rmse_c=stats.mean(cd), channel_contrast_c=stats.mean(cc),
            change_in_contrast_c=stats.mean(changes), median_change_c=stats.median(changes),
            profiles_change_below_zero=sum(v < 0 for v in changes),
            deletion_min_change_c=min(deletion), deletion_max_change_c=max(deletion), smoke_only=smoke))
    receipt = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   paired_profiles=len(profiles), arm_rows=len(ledger),
                   cohorts=len(cohorts), smoke_only=smoke,
                   contrast="(residual - matched direct) channel-wise minus ambient, degrees Celsius",
                   scope="Descriptive, equal-profile means after arm-specific refits; deletion is without refitting.")
    return {"cohorts.csv": csv_text(cohorts), "profiles.csv": csv_text(profiles),
            "receipt.json": json.dumps(receipt, indent=2) + "\n"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ARCHIVE, help="failure-profile-corrections.csv")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path, help="new result directory")
    group.add_argument("--check", action="store_true", help="recalculate and compare to all saved v013 contrast files")
    args = parser.parse_args()
    files = derive(args.source)
    if args.check:
        for name, value in files.items():
            if (FROZEN / name).read_bytes() != value.encode("utf-8"):
                raise ValueError(f"Frozen v013 contrast result differs: {name}")
        # The other historical analysis aggregated its own prediction-error arrays,
        # providing an independent representation of the same within-profile contrasts.
        companion = ROOT / "reference/v007/analysis/initialization-whole-profile.csv"
        with companion.open(encoding="utf-8", newline="") as stream:
            index = {(r["dataset"], r["role"], r["profile"], r["method"]): r for r in csv.DictReader(stream)}
        for row in csv.DictReader(io.StringIO(files["profiles.csv"])):
            key = (row["dataset"], row["role"], row["profile"])
            for column, cross_column in (("ambient_residual_minus_direct_c", "original_rmse"),
                                         ("channel_residual_minus_direct_c", "own_rmse")):
                r, d = [index[(*key, model)] for model in ("linear_residual", "direct200")]
                if not (r["n"] == d["n"] == row["n"] and math.isclose(
                        float(row[column]), float(r[cross_column]) - float(d[cross_column]),
                        rel_tol=0, abs_tol=1e-10)):
                    raise ValueError(f"Independent historical ledger mismatch: {key}/{column}")
        print("PASS: all five cohorts, 36 paired profiles, source hash and deletion summaries match.")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        for name, value in files.items():
            (args.output / name).write_bytes(value.encode("utf-8"))
        print(f"Wrote {len(files)} files to {args.output}")


if __name__ == "__main__":
    main()
