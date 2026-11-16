"""Bounded CPU tests; optional saved-artifact parity never launches old runners.

Set PMSM_HISTORICAL_ROOT to an existing campaign to enable read-only parity.
Historical data and fitted artifacts are not required or distributed by this test.
"""
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from threadpoolctl import threadpool_limits
from pmsm_repro import features as f, models as m, metrics as mm


class NumericTests(unittest.TestCase):
    def test_causal_initialization(self):
        t = np.arange(403, dtype=float)
        op = np.column_stack((2 + .1 * np.sin(t), 3 + 0*t, 5 + 0*t, 1 + 0*t,
                              80 + t, 35 + t/80, 20 + 0*t))
        raw = dict(id=1, operating=op, y=40 + t/20)
        original = f.make_record("ipmsm", raw)
        own = f.make_record("ipmsm", raw, "own_boundary")
        first = f.make_record("ipmsm", raw, "first_load")
        for j, span in enumerate(f.SPANS):
            decay = (1 - 2/(span+1)) ** t
            np.testing.assert_allclose(own["boundary"][:, 2*j+1] - original["boundary"][:, 2*j+1], 15*decay, atol=2e-12)
            np.testing.assert_allclose(first["heat"][:, 3*j] - original["heat"][:, 3*j], 13*decay, atol=2e-12)
        np.testing.assert_array_equal(own["heat"], original["heat"])
        prefix = f.make_record("ipmsm", dict(raw, operating=op[:350], y=raw["y"][:350]), "own_boundary")
        np.testing.assert_array_equal(prefix["boundary"], own["boundary"][:350])
        np.testing.assert_array_equal(f.train_index(own), np.arange(300, 403, 5))
        np.testing.assert_array_equal(f.evaluation_index(own), np.arange(300, 403))

    def test_rwth_timestamp_recurrence(self):
        op = np.tile([2., 60., 3., 10.], (7, 1))
        raw = dict(id=2, operating=op, time=np.arange(7, dtype=float), y=np.arange(7, dtype=float))
        rec = f.make_record("rwth", raw, dt_ref=1.)
        np.testing.assert_array_equal(rec["X33"][0, 8:13], np.full(5, 10.))
        np.testing.assert_array_equal(rec["X33"][0, 13:], np.zeros(20))
        np.testing.assert_allclose(rec["X33"][6, 13:18], 4*(1-(1-2/(np.asarray(f.SPANS)+1))**6), atol=1e-13)
        changed = f.make_record("rwth", dict(raw, time=raw["time"]*17), dt_ref=17)
        np.testing.assert_array_equal(changed["X33"], rec["X33"])
        with self.assertRaises(ValueError):
            f.make_record("rwth", raw, "own_boundary", dt_ref=1.)

    def test_weights_and_metrics(self):
        sumone = f.balanced_weights([2, 8])
        self.assertAlmostEqual(sumone[:2].sum(), .5)
        self.assertAlmostEqual(sumone[2:].sum(), .5)
        row = mm.profile_metrics([0, 0], [-2, 4])
        self.assertEqual(row["under_rmse"], np.sqrt(2))
        self.assertEqual(row["over_mae"], 2.)
        other = mm.profile_metrics([0], [0])
        agg = mm.aggregate_profiles([row, other])
        self.assertAlmostEqual(agg["macro_rmse"], np.sqrt(10)/2)
        self.assertAlmostEqual(agg["pooled_rmse"], np.sqrt(20/3))
        identity = mm.ensemble_decomposition([1, 3, 9], [2, 0, 8], [0, 4, 12])
        self.assertAlmostEqual(identity["delta_mse"], identity["expert_quality_term"]+identity["diversity_term"])

    def test_study_config_fits_and_preserves_overrides(self):
        """The real complete study settings must overlay defaults without collisions."""
        study = json.loads((Path(__file__).resolve().parents[1]/"configs/study.json").read_text())
        rng = np.random.RandomState(52)
        for ds, spec in study["datasets"].items():
            n = 410 if ds != "rwth" else 90
            records = {}
            for pid in (1, 2):
                op = rng.normal(size=(n, 4 if ds == "rwth" else 7))
                records[pid] = f.make_record(ds, dict(id=pid, operating=op,
                    y=30+.3*op[:, 0], time=np.arange(n, dtype=float)), dt_ref=1.)
            with threadpool_limits(limits=1):
                bundle = m.fit_cpu(records, [1, 2], ds, spec, smoke=True)
            for name in ("selected_direct", "matched_direct", "capacity_direct"):
                expected = spec["cpu"][name]
                actual = bundle["metadata"]["configurations"][name]
                for key, value in expected.items():
                    self.assertEqual(actual[key], min(3, value) if key == "max_iter" else value)
                self.assertEqual(bundle["boosters"][name].n_iter_, 3)
            self.assertEqual(bundle["priors"]["linear"]["lam"], spec["cpu"]["lambda"])

    def test_smoke_fit_lazy_prediction_and_storage(self):
        rng = np.random.RandomState(31)
        for ds in ("ipmsm", "rwth"):
            records = {}
            for pid in (1, 2):
                n = 420 if ds == "ipmsm" else 100
                op = rng.normal(size=(n, 7 if ds == "ipmsm" else 4))
                y = 30 + .3*op[:, 0] + .1*np.arange(n)/n
                raw = dict(id=pid, operating=op, y=y, time=np.arange(n, dtype=float))
                records[pid] = f.make_record(ds, raw, dt_ref=1.)
            calls = []
            with threadpool_limits(limits=1):
                bundle = m.fit_cpu(records, [1, 2], ds, {}, smoke=True, check=lambda: calls.append(1))
                values = m.predict_cpu(bundle, records[1])
                np.testing.assert_array_equal(values["linear_half"], .5*(values["selected_direct"]+values["linear_residual"]))
                self.assertEqual(len(calls), 8)
                # A prior-only request must not call a booster at all.
                original = bundle["boosters"]
                bundle["boosters"] = {}
                np.testing.assert_array_equal(m.predict_method(bundle, records[1], "linear_prior"), values["linear_prior"])
                bundle["boosters"] = original
                with tempfile.TemporaryDirectory() as td:
                    folder = Path(td)/"model"
                    m.save_cpu(bundle, folder)
                    restored = m.load_cpu(folder)
                    np.testing.assert_array_equal(m.predict_method(restored, records[1], "linear_residual"), values["linear_residual"])
                    manifest = json.loads((folder/"manifest.json").read_text())
                    del manifest["files"]["model.json"]
                    (folder/"manifest.json").write_text(json.dumps(manifest))
                    with self.assertRaises(ValueError):
                        m.load_cpu(folder)
            with threadpool_limits(limits=1):
                limited = m.fit_cpu(records, [1, 2], ds,
                                    {"fit_components": ["selected_direct", "residual"]}, smoke=True)
                self.assertEqual(set(limited["priors"]), {"linear"})
                self.assertEqual(set(m.predict_cpu(limited, records[1])),
                                 {"selected_direct", "linear_prior", "linear_residual", "linear_half", "prior_half"})


HISTORICAL = os.environ.get("PMSM_HISTORICAL_ROOT")


@unittest.skipUnless(HISTORICAL, "Optional historical archive path not supplied")
class HistoricalParity(unittest.TestCase):
    def setUp(self):
        self.root = Path(HISTORICAL).resolve()

    def read(self, rel):
        return json.loads((self.root/rel).read_text(encoding="utf-8-sig"))

    def pinned(self, entry):
        path = (self.root/entry["path"]).resolve()
        path.relative_to(self.root)
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                h.update(block)
        self.assertEqual(h.hexdigest(), entry["sha256"])
        return path

    def arrays(self, entry):
        with np.load(self.pinned(entry), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def assert_archived_features(self, actual, expected):
        # Original scalar-feature parity bound. Windows/Linux hypot/filter paths
        # can differ by a few FP64 ulps; cross-platform bit identity is not claimed.
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=2e-12)

    def test_saved_ipmsm_linear_solve(self):
        """Replay two small ridge solves; no boosting or historical stage execution."""
        config = self.read("linear-followup-20261111/configs/ipmsm-own_boundary-initialization-v002.json")
        records = {}
        for entry in config["fit_cache"]:
            saved = self.arrays(entry)
            records[entry["id"]] = f.make_record("ipmsm", dict(id=entry["id"],
                operating=saved["operating"], y=saved["target"]), "own_boundary")
        for family, path in (("linear", "linear-followup-20261111/runs/ipmsm-own_boundary-initialization-fit-v002/linear_model.json"),
                             ("history", "linear-followup-20261111/runs/ipmsm-own-controls-cpu-v001/free_prior.json")):
            expected = self.read(path)
            with threadpool_limits(limits=1):
                actual = m.fit_prior(records, config["fit_ids"], family, expected["lam"])
            np.testing.assert_allclose(actual["coef"], expected["coef"], atol=2e-10, rtol=1e-10)
            for key in ("qscale", "y_mid", "y_scale"):
                self.assert_archived_features(actual["scale"][key], expected["scale"][key])

    def test_legacy_saved_features_models_and_metrics(self):
        for ds in ("ipmsm", "paderborn"):
            for arm in ("own_boundary", "first_load"):
                run = f"linear-followup-20261111/runs/{ds}-{arm}-initialization-fit-v002"
                lock = self.read(run+"/prediction_lock.json")
                entry = lock["records"][0]
                old = self.arrays(entry["source_features"])
                saved = self.arrays(entry["prediction"])
                rec = f.make_record(ds, dict(id=entry["id"], operating=old["operating"]), arm)
                for key in ("boundary", "heat", "instant"):
                    self.assert_archived_features(rec[key], saved[key])
                original = f.make_record(ds, dict(operating=old["operating"]))
                for key in ("boundary", "heat", "instant", "ewma"):
                    self.assert_archived_features(original[key], old[key])
                prior = self.read(run+"/linear_model.json")
                prior.update(family="linear", dataset=ds)
                ix = saved["index"]
                self.assert_archived_features(f.boost_features(rec, prior["scale"])[ix], saved["features"])
                np.testing.assert_allclose(m.predict_prior(prior, rec)[ix], saved["prediction__linear_prior"], atol=1e-10, rtol=1e-12)
                # Trusted, hash-pinned local fitted model: bounded rows only, no fit.
                model_lock = self.read(run+"/model_lock.json")
                meta = self.read(model_lock["boosters"]["residual"]["path"])
                with self.pinned(meta["pickle"]).open("rb") as stream:
                    booster = pickle.load(stream)
                rows = np.array([0, 1, len(ix)//2, len(ix)-1])
                with threadpool_limits(limits=1):
                    predicted = m.predict_prior(prior, rec)[ix[rows]] + booster.predict(saved["features"][rows])
                np.testing.assert_allclose(predicted, saved["prediction__linear_residual"][rows], atol=1e-10, rtol=1e-12)
            run = f"linear-followup-20261111/runs/{ds}-own-controls-cpu-v001"
            free = self.read(run+"/free_prior.json")
            free.update(family="history", dataset=ds)
            # Same first profile from the channel-wise arm.
            lock = self.read(f"linear-followup-20261111/runs/{ds}-own_boundary-initialization-fit-v002/prediction_lock.json")
            entry = lock["records"][0]
            saved = self.arrays(entry["prediction"])
            rec = dict(dataset=ds, **{k:saved[k] for k in ("boundary", "heat", "instant")})
            cpulock = self.read(run+"/prediction_lock.json")
            pred = self.arrays(cpulock["records"][0]["prediction"])
            np.testing.assert_allclose(m.predict_prior(free, rec)[saved["index"]], pred["prediction__free_prior"], atol=1e-10, rtol=1e-12)
            y = self.arrays(entry["truth"])["target"]
            results = self.read(f"linear-followup-20261111/runs/{ds}-own-controls-evaluate-v001/results.json")
            expected = results["per_profile"][0]["methods"]["linear_residual"]
            observed = mm.profile_metrics(y, pred["prediction__linear_residual"])
            for key in expected:
                self.assertAlmostEqual(observed[key], expected[key], places=11)

    def test_rwth_saved_features_and_prior(self):
        manifest = self.read("followup-risk-20261110/runs/rwth-prepare-v002/manifest.json")
        entry = manifest["records"][0]
        saved = self.arrays(entry["features"])
        rec = f.make_record("rwth", dict(id=entry["id"], operating=saved["operating"],
                             y=saved["target"], time=saved["time"]), dt_ref=manifest["dt_ref"])
        self.assert_archived_features(rec["X33"], saved["X33"])
        priors = self.read("followup-risk-20261110/runs/rwth-cpu-final-v002/priors.json")
        pred_lock = self.read("followup-risk-20261110/runs/rwth-cal-cpu-predict-v002/prediction_lock.json")
        entry = pred_lock["records"][0]
        held = self.arrays(entry["source_features"])
        prediction = self.arrays(entry)
        rec = f.make_record("rwth", dict(id=entry["id"], operating=held["operating"],
                             time=held["time"]), dt_ref=manifest["dt_ref"])
        self.assert_archived_features(rec["X33"], held["X33"])
        for name, family in (("linear33", "linear"), ("free25", "history")):
            prior = dict(priors[name], family=family, dataset="rwth")
            saved_key = "prediction__linear33_prior" if family == "linear" else "prediction__unconstrained25_prior"
            np.testing.assert_allclose(m.predict_prior(prior, rec), prediction[saved_key], atol=1e-10, rtol=1e-12)
        run = "followup-risk-20261110/runs/rwth-cpu-final-v002/linear33_insample_boost"
        meta = self.read(run+"/model.json")
        with self.pinned(dict(path=run+"/model.pkl", sha256=meta["files"]["model.pkl"])).open("rb") as stream:
            booster = pickle.load(stream)
        rows = np.array([0, 1, len(held["time"])//2, len(held["time"])-1])
        prior = dict(priors["linear33"], family="linear", dataset="rwth")
        with threadpool_limits(limits=1):
            actual = m.predict_prior(prior, rec)[rows] + booster.predict(f.boost_features(rec, prior["scale"])[rows])
        np.testing.assert_allclose(actual, prediction["prediction__linear33_insample_boost"][rows], atol=1e-10, rtol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
