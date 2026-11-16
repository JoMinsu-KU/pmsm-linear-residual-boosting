"""Focused checks for the public, read-only v007 analysis entry points."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


failure = load_script("analyze_failure")
initialization = load_script("analyze_initialization")
ANALYSES = (failure.run_diagnostics, initialization.compute_from_run)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def seal(run, name):
    """Create a small legitimate stage receipt, using the pipeline contract."""
    folder = run / name
    binding = {"run": json.loads((run / "run.json").read_text())["key"], "stage": name, "extra": None}
    key = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    files = [{"path": p.relative_to(folder).as_posix(),
              "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(folder.rglob("*")) if p.is_file() and p.name != "complete.json"]
    write_json(folder / "complete.json", {"input_key": key, "files": files})


def portable_fixture(run):
    """Tiny completed cohorts; all predictors improve, so no failure plot qualifies."""
    binding = {"schema": "portable-study-run-v2", "mode": "smoke", "datasets": ["ipmsm", "paderborn"],
               "numerical_runtime": {}, "device": "cpu", "files": {}}
    run_key = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write_json(run / "run.json", {"key": run_key, **binding})
    roles = {"ipmsm": {"fit": [0], "calibration": [1], "pilot": [2], "confirmation": [3]},
             "paderborn": {"fit": [10], "development": [11], "confirmation": [12]}}
    config = {"spans": [16, 64, 256, 1024, 4096],
              "datasets": {ds: {"roles": values, "warmup": 300, "train_stride": 5, "evaluation_stride": 1}
                           for ds, values in roles.items()}}
    write_json(run / "report/metadata.json", {"configuration": config})
    seal(run, "report")
    for ds, partitions in roles.items():
        cache = run / "cache" / ds
        cache.mkdir(parents=True)
        profiles = []
        for ids in partitions.values():
            for pid in ids:
                y = np.linspace(20., 30., 312)
                np.savez(cache / f"profile_{pid}.npz", y=y, operating=np.ones((312, 6)))
                profiles.append({"id": pid, "path": f"profile_{pid}.npz"})
        write_json(cache / "dataset.json", {"dataset": ds, "roles": partitions, "profiles": profiles})
        seal(run, f"cache/{ds}")
        for arm in ("original", "own_boundary"):
            model = {"dataset": ds, "metadata": {"arm": arm, "fit_ids": partitions["fit"],
                "configurations": {name: {"trees": 2} for name in ("residual", "matched_direct", "selected_direct")}}}
            write_json(run / f"models/{ds}/{arm}/cpu/cpu/model.json", model)
            seal(run, f"models/{ds}/{arm}/cpu")
            for role, ids in partitions.items():
                if role == "fit":
                    continue
                folder = run / f"evaluation/{ds}/{arm}/{role}"
                folder.mkdir(parents=True)
                for pid in ids:
                    target = y[300:]
                    shift = .25 if arm == "own_boundary" else .5
                    predictions = {"prediction__" + method: target + error + shift
                                   for method, error in (("linear_prior", 2.), ("linear_residual", .5),
                                                         ("direct200", 1.), ("d9", 1.5))}
                    np.savez(folder / f"profile_{pid}.npz", index=np.arange(300, 312),
                             target=target, **predictions)
            seal(run, f"evaluation/{ds}/{arm}")
    return run


class ReviewArithmeticTests(unittest.TestCase):
    def test_residual_decomposition_and_real_overshoot(self):
        y = np.zeros(3)
        prior = np.array([2., -2., 0.])
        correction = np.array([-1., 5., 2.])
        z = {"prediction__linear_prior": prior,
             "prediction__linear_residual": prior + correction,
             "prediction__direct200": np.ones(3), "prediction__d9": np.ones(3)}
        row = failure.metric_rows("ipmsm", "own_boundary", "confirmation", 1, z, y)
        self.assertAlmostEqual(row["correction_cross_term_c2"], -8.)
        self.assertAlmostEqual(row["correction_energy_c2"], 10.)
        self.assertAlmostEqual(row["residual_minus_prior_mse_c2"], 2.)
        self.assertAlmostEqual(row["opposing_but_overshooting_percent"], 100. / 3)
        self.assertAlmostEqual(row["rowwise_squared_error_reduced_percent"], 100. / 3)
        self.assertLess(row["identity_absolute_discrepancy"], 1e-14)

    def test_uneven_fifths_preserve_every_endpoint_and_weighted_mse(self):
        ix = np.arange(300, 313)
        y = np.zeros(13)
        original = {m: np.arange(1., 14.) for m in initialization.METHODS}
        own = {m: original[m] / 2 for m in initialization.METHODS}
        segments, pairs, whole = initialization.profile_metrics_pair("ipmsm", "pilot", 1, ix, y, original, own)
        rows = [r for r in pairs if r["method"] == "linear_residual"]
        self.assertEqual([r["n"] for r in rows], [2, 3, 2, 3, 3])
        covered = [i for r in rows for i in range(r["first_source_index"], r["last_source_index"] + 1)]
        self.assertEqual(covered, ix.tolist())
        self.assertEqual(len(segments), 30)
        self.assertAlmostEqual(sum(r["original_mse"] * r["n"] for r in rows) / 13, whole[0]["original_mse"])
        self.assertAlmostEqual(whole[0]["own_mse"], whole[0]["original_mse"] / 4)
        with self.assertRaisesRegex(ValueError, "Contiguous"):
            initialization.profile_metrics_pair("ipmsm", "pilot", 1, ix * 2, y, original, own)

    def test_macro_rmse_gives_equal_weight_to_unequal_profile_lengths(self):
        pairs, whole = [], []
        for ds, role in initialization.COHORTS:
            for pid, n, error in ((1, 10, 1.), (2, 20, 3.)):
                _, p, h = initialization.profile_metrics_pair(ds, role, pid, np.arange(300, 300 + n),
                    np.zeros(n), {m: np.full(n, error) for m in initialization.METHODS},
                    {m: np.zeros(n) for m in initialization.METHODS})
                pairs.extend(p)
                whole.extend(h)
        summaries, _ = initialization.aggregate_pairs(pairs, whole)
        self.assertEqual(len(summaries), 75)
        self.assertAlmostEqual(summaries[0]["original_macro_rmse"], 2.)
        self.assertAlmostEqual(summaries[0]["macro_difference"], -2.)
        self.assertAlmostEqual(summaries[0]["original_pooled_rmse"], np.sqrt(19. / 3))


class ReviewInputTests(unittest.TestCase):
    def test_cli_requires_explicit_run_and_output(self):
        for module in (failure, initialization):
            for args in ([], ["--run-dir", "unused"], ["--output", "unused"]):
                with self.subTest(script=module.__name__, args=args):
                    result = subprocess.run([sys.executable, module.__file__, *args], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("required", result.stderr)

    def test_existing_and_inside_run_outputs_rejected_before_input_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            for analyze in ANALYSES:
                with self.subTest(script=analyze.__module__):
                    with self.assertRaises(FileExistsError):
                        analyze(run, root)
                    with self.assertRaisesRegex(ValueError, "outside"):
                        analyze(run, run / "new")
                    self.assertFalse((run / "new").exists())

    def test_missing_stage_and_changed_file_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = portable_fixture(root / "run")
            receipt = run / "report/complete.json"
            saved = receipt.read_bytes()
            receipt.unlink()
            for analyze in ANALYSES:
                with self.assertRaises(FileNotFoundError):
                    analyze(run, root / "missing-output")
                self.assertFalse((root / "missing-output").exists())
            receipt.write_bytes(saved)
            with (run / "report/metadata.json").open("a", encoding="utf-8") as stream:
                stream.write(" ")
            for analyze in ANALYSES:
                with self.assertRaisesRegex(ValueError, "hash|changed"):
                    analyze(run, root / "hash-output")
                self.assertFalse((root / "hash-output").exists())

    def test_wrong_stage_binding_and_unlisted_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = portable_fixture(root / "run")
            receipt = run / "report/complete.json"
            data = json.loads(receipt.read_text())
            data["input_key"] = "another-run"
            write_json(receipt, data)
            for analyze in ANALYSES:
                with self.assertRaisesRegex(ValueError, "binding|another run"):
                    analyze(run, root / "output")
            seal(run, "report")
            (run / "report/unlisted.txt").write_text("unexpected")
            for analyze in ANALYSES:
                with self.assertRaisesRegex(ValueError, "inventory"):
                    analyze(run, root / "output")

    def test_initialization_settings_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = portable_fixture(root / "run")
            path = run / "models/ipmsm/own_boundary/cpu/cpu/model.json"
            model = json.loads(path.read_text())
            model["metadata"]["configurations"]["residual"]["trees"] = 3
            write_json(path, model)
            seal(run, "models/ipmsm/own_boundary/cpu")
            with self.assertRaisesRegex(ValueError, "settings mismatch"):
                initialization.compute_from_run(run, root / "output")
            self.assertFalse((root / "output").exists())

    def test_smoke_mode_cannot_be_relabelled_as_paper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = portable_fixture(root / "run")
            path = run / "run.json"
            manifest = json.loads(path.read_text())
            manifest["mode"] = "paper"
            write_json(path, manifest)
            for analyze in ANALYSES:
                with self.assertRaisesRegex(ValueError, "Run binding mismatch"):
                    analyze(run, root / "output")
                self.assertFalse((root / "output").exists())

    def test_unsupported_warmup_or_strides_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = portable_fixture(root / "run")
            path = run / "report/metadata.json"
            for name, changed in (("warmup", 301), ("train_stride", 6), ("evaluation_stride", 2)):
                metadata = json.loads(path.read_text())
                original = metadata["configuration"]["datasets"]["ipmsm"][name]
                metadata["configuration"]["datasets"]["ipmsm"][name] = changed
                write_json(path, metadata)
                seal(run, "report")
                for analyze in ANALYSES:
                    with self.assertRaisesRegex(ValueError, "Unsupported warmup/stride"):
                        analyze(run, root / "output")
                    self.assertFalse((root / "output").exists())
                metadata["configuration"]["datasets"]["ipmsm"][name] = original
                write_json(path, metadata)

    def test_cohort_missing_from_both_arms_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = portable_fixture(root / "run")
            for arm in ("original", "own_boundary"):
                (run / f"evaluation/ipmsm/{arm}/pilot/profile_2.npz").unlink()
                seal(run, f"evaluation/ipmsm/{arm}")
            for analyze in ANALYSES:
                with self.assertRaisesRegex(ValueError, "profile set|complete cohort"):
                    analyze(run, root / "output")
                self.assertFalse((root / "output").exists())

    def test_smoke_analysis_labels_and_no_fabricated_failure_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = portable_fixture(root / "run")
            with contextlib.redirect_stdout(io.StringIO()):
                result = failure.run_diagnostics(run, root / "failure")
            self.assertTrue(result["smoke_only"])
            self.assertFalse(result["historical_fallback_used"])
            self.assertEqual(result["profiles"], 10)
            self.assertEqual(result["selected_examples"], [])
            self.assertIn("No qualifying", result["figure_unavailable_reason"])
            self.assertFalse((root / "failure/figures").exists())
            rows, result = initialization.compute_from_run(run, root / "initialization")
            self.assertTrue(result["smoke_only"])
            self.assertFalse(result["historical_fallback_used"])
            self.assertEqual(result["paired_profiles"], 5)
            self.assertEqual(len(rows), 75)
            self.assertTrue(all(row["smoke_only"] for row in rows))
            self.assertTrue(all(check["same_settings"] for check in result["model_settings_checks"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
