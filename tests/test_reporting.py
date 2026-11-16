"""Evidence-provenance and scientific-aggregation tests for the report exporter."""
from __future__ import annotations

import copy
import csv
import math
from pathlib import Path
import tempfile
import unittest

from pmsm_repro.reporting import (
    generate_report, IncompleteReportError, MAIN_COHORTS, MAIN_METHODS,
    REQUIRED_SETTING_LABELS, method_label,
)


def fixture():
    """Small invented inputs are confined to tests and never shipped as results."""
    accuracy, profiles, timing = [], [], []
    methods = [m for m in MAIN_METHODS if m != "mlp_mean"] + ["mse_mlp_11", "mse_mlp_12", "mse_mlp_13"]
    cohorts = list(MAIN_COHORTS) + [("rwth", "calibration"), ("rwth", "confirmation")]
    for dataset, role in cohorts:
        for arm in ("original", "own_boundary"):
            for k, method in enumerate(methods):
                scale = (1 if arm == "original" else .9) * (1 + k / 5)
                errors = [1 * scale, 3 * scale]
                accuracy.append(dict(dataset=dataset, role=role, arm=arm, method=method, profiles=2, n=30,
                                     macro_rmse=sum(errors)/2, pooled_rmse=math.sqrt((10*errors[0]**2+20*errors[1]**2)/30),
                                     macro_under_rmse=.5*scale, macro_over_mae=.4*scale, macro_max_under=3*scale))
                for pid, n, rmse, maximum in [("first", 10, errors[0], 2*scale), ("second", 20, errors[1], 4*scale)]:
                    profiles.append(dict(dataset=dataset, role=role, arm=arm, method=method, profile=pid,
                                         n=n, mse=rmse**2, rmse=rmse, under_rmse=.5*scale, over_mae=.4*scale, max_under=maximum))
    for k, method in enumerate([m for m in methods if not m.startswith("mse_mlp_")] + ["mse_mlp_11"]):
        timing.append(dict(dataset="paderborn", arm="own_boundary", method=method, profile_id="first",
                           median_us=10+k, p95_us=20+k, artifact_bytes=1000+k))
    metadata = {"execution_mode": "synthetic test fixture", "expected_neural_seed_count": 3,
                "datasets": {ds: dict(training_profiles=["train"], total_profiles=5, warmup_exclusion=0) for ds in ["ipmsm", "paderborn", "rwth"]},
                "model_settings": {label: "Test setting (not experimental evidence)" for label in REQUIRED_SETTING_LABELS}}
    return accuracy, profiles, timing, metadata


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class ReportingTests(unittest.TestCase):
    def test_generated_metrics_drive_all_displays_and_seed_pairing(self):
        accuracy, profiles, timing, metadata = fixture()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            result = generate_report(out, accuracy, profiles, timing, metadata)
            self.assertEqual(result["status"], "COMPLETE")
            self.assertFalse(result["historical_fallback_used"])
            self.assertEqual(len(list((out/"tables").glob("table*.csv"))), 7)
            self.assertEqual(len(list((out/"tables").glob("table*.tex"))), 7)
            self.assertEqual(len(list((out/"figures").glob("*.png"))), 4)
            self.assertEqual(len(list((out/"figures").glob("*.pdf"))), 4)
            table = read_csv(out/"tables/table4.csv")
            mean_row = next(r for r in table if r["Method"] == method_label("mlp_mean"))
            seeds = [r for r in accuracy if (r["dataset"],r["role"],r["arm"]) == ("paderborn","development","own_boundary") and r["method"].startswith("mse_mlp_")]
            self.assertAlmostEqual(float(mean_row["Paderborn development"]),sum(r["macro_rmse"] for r in seeds)/3,places=6)
            points = read_csv(out/"figures/figure2-data.csv")
            neural = next(p for p in points if p["cohort"] == "Development" and p["method"] == "Neural network (seed 11)")
            timed_accuracy = next(r["macro_rmse"] for r in seeds if r["method"] == "mse_mlp_11")
            self.assertEqual(float(neural["macro_rmse"]),timed_accuracy)
            self.assertNotEqual(float(neural["macro_rmse"]),float(mean_row["Paderborn development"]))
            table5 = {r["Quantity"]:r for r in read_csv(out/"tables/table5.csv")}
            method = method_label("linear_residual")
            self.assertGreater(float(table5["Worst maximum underestimation"][method]),float(table5["Mean maximum underestimation"][method]))
            changes = read_csv(out/"figures/figure4-data.csv")
            self.assertEqual(len(changes), 15)
            self.assertTrue(all(abs(float(x["change_percent"])+10)<1e-10 for x in changes))
            text = (out/"tables/table4.csv").read_text(encoding="utf-8")
            self.assertNotIn("d9",text)
            self.assertNotIn("linear_half",text)

    def test_no_fallback_for_missing_smoke_evidence(self):
        accuracy, profiles, timing, metadata = fixture()
        accuracy = [r for r in accuracy if r["dataset"] == "ipmsm"]
        profiles = [r for r in profiles if r["dataset"] == "ipmsm"]
        metadata["execution_mode"] = "smoke"
        with tempfile.TemporaryDirectory() as td:
            strict_out = Path(td)/"strict"
            with self.assertRaises(IncompleteReportError):
                generate_report(strict_out,accuracy,profiles,[],metadata)
            self.assertFalse(strict_out.exists())
            out = Path(td)/"partial"
            report = generate_report(out,accuracy,profiles,[],metadata,strict=False)
            self.assertEqual(report["status"],"PARTIAL_EXPLICIT_MISSING_INPUTS")
            self.assertFalse(report["historical_fallback_used"])
            self.assertTrue(report["missing_inputs"])
            self.assertEqual(read_csv(out/"figures/figure2-data.csv"),[])
            self.assertEqual(len(list((out/"figures").glob("*.pdf"))),4)
            table4=read_csv(out/"tables/table4.csv")
            self.assertTrue(all(r["Paderborn development"] == "" for r in table4))

    def test_reference_mode_is_explicit_and_uses_supplied_values(self):
        accuracy, profiles, timing, metadata = fixture()
        row = next(r for r in accuracy if (r["dataset"],r["role"],r["arm"],r["method"]) == ("paderborn","confirmation","own_boundary","linear_residual"))
        row["macro_rmse"] = 9.876543
        with tempfile.TemporaryDirectory() as td:
            report=generate_report(td,accuracy,profiles,timing,metadata,mode="reference_snapshot")
            self.assertEqual(report["mode"],"reference_snapshot")
            table = read_csv(Path(td)/"tables/table4.csv")
            value=next(r["Paderborn historical evaluation"] for r in table if r["Method"]==method_label("linear_residual"))
            self.assertEqual(value,"9.876543")
            self.assertIn("reference snapshot; not a new experimental run",(Path(td)/"tables/table4.tex").read_text())

    def test_unidentified_timed_seed_and_unpaired_profiles_rejected(self):
        accuracy, profiles, timing, metadata = fixture()
        timing[-1]["method"]="mlp_mse_seed1"
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,"timed_neural_seed"):
                generate_report(td,accuracy,profiles,timing,metadata)
        accuracy, profiles, timing, metadata = fixture()
        profiles = [r for r in profiles if not (r["dataset"]=="paderborn" and r["arm"]=="own_boundary" and r["role"]=="development" and r["method"]=="direct200" and r["profile"]=="first")]
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,"exactly paired"):
                generate_report(td,accuracy,profiles,timing,metadata)
        accuracy, profiles, timing, metadata = fixture()
        timing[-1]["profile_id"] = "a different workload"
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,"same fixed profile"):
                generate_report(td,accuracy,profiles,timing,metadata)

    def test_contradictory_neural_mean_and_nonfinite_inputs_rejected(self):
        accuracy, profiles, timing, metadata = fixture()
        mean=copy.deepcopy(accuracy[0]);mean["method"]="mlp_mean";mean["macro_rmse"]=999.;accuracy.append(mean)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,"neural mean disagrees"):
                generate_report(td,accuracy,profiles,timing,metadata)
        accuracy, profiles, timing, metadata=fixture();accuracy[0]["macro_rmse"]=float("nan")
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,"finite"):
                generate_report(td,accuracy,profiles,timing,metadata)


if __name__ == "__main__":
    unittest.main()
