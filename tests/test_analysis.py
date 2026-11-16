import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from pmsm_repro.analysis import summarize_diagnostics


def fixture(count=3):
    profiles, segments, diagnostics = [], [], []
    for pid in range(1, count+1):
        for arm in ("original", "own_boundary"):
            base = dict(dataset="paderborn", arm=arm, role="development", profile=pid)
            # B=(1,2,3), signed change=(1,2,3): both associations equal+1.
            values = dict(d9=10., linear_half=8., direct200=9.,
                          linear_residual=10.+(pid if arm == "own_boundary" else 0))
            for method, value in values.items():
                profiles.append(dict(base, method=method, n=50, rmse=value))
                for fifth in range(1,6):
                    # Unequal profile segment lengths must not weight the macro mean.
                    val = 10.-pid*fifth if method == "linear_half" else 10.
                    segments.append(dict(base, method=method, fifth=fifth, n=10*pid,
                                         rmse=abs(val), first_source_index=10*(fifth-1), last_source_index=10*fifth-1))
            diagnostics.append(dict(base,boundary_history_rms_difference=float(pid)))
    return profiles,segments,diagnostics


class AnalysisTests(unittest.TestCase):
    def test_equal_profile_fifths_and_signed_correlations(self):
        data=fixture()
        result=summarize_diagnostics(*data)
        row=next(r for r in result["temporal_fifth_contrasts"] if r["arm"]=="own_boundary" and r["fifth"]==1 and r["left"]=="linear_half")
        self.assertEqual(row["macro_difference_c"],-2.)
        self.assertEqual(row["n"],60)
        corr=result["initialization_associations"][0]
        self.assertAlmostEqual(corr["pearson_signed_change"]["value"],1.)
        self.assertAlmostEqual(corr["spearman_signed_change"]["value"],1.)
        self.assertEqual([p["percent_change"] for p in corr["per_profile"]],[10.,20.,30.])
        json.dumps(result,allow_nan=False)

    def test_missing_pairs_are_not_silently_dropped(self):
        profiles,segments,diagnostics=fixture()
        segments=[r for r in segments if not (r["arm"]=="own_boundary" and r["profile"]==3 and r["method"]=="d9" and r["fifth"]==1)]
        diagnostics=[r for r in diagnostics if not(r["arm"]=="original" and r["profile"]==3)]
        result=summarize_diagnostics(profiles,segments,diagnostics)
        row=next(r for r in result["temporal_fifth_contrasts"] if r["arm"]=="own_boundary" and r["fifth"]==1 and r["left"]=="linear_half")
        self.assertIsNone(row["macro_difference_c"])
        self.assertEqual(row["profiles"],2)
        self.assertEqual(row["expected_profiles"],3)
        corr=result["initialization_associations"][0]
        self.assertIsNone(corr["pearson_signed_change"]["value"])
        self.assertEqual(corr["pearson_signed_change"]["null_reason"],"missing_initialization_pairs_or_B")

    def test_ties_small_samples_and_constant_values(self):
        result=summarize_diagnostics(*fixture(1))["initialization_associations"][0]
        self.assertEqual(result["pearson_signed_change"]["null_reason"],"fewer_than_two_profiles")
        result=summarize_diagnostics(*fixture(2))["initialization_associations"][0]
        self.assertAlmostEqual(result["pearson_signed_change"]["value"],1.)
        self.assertEqual(result["spearman_signed_change"]["null_reason"],"fewer_than_three_profiles")
        profiles,segments,diagnostics=fixture()
        for d in diagnostics:d["boundary_history_rms_difference"]=1.
        result=summarize_diagnostics(profiles,segments,diagnostics)["initialization_associations"][0]
        self.assertEqual(result["pearson_signed_change"]["null_reason"],"zero_variance")
        self.assertEqual(result["spearman_signed_change"]["null_reason"],"constant_profile_rank")

    def test_mismatched_endpoints_and_duplicate_rows_rejected(self):
        profiles,segments,diagnostics=fixture()
        segments[0]["n"]+=1
        with self.assertRaisesRegex(ValueError,"endpoints differ"):
            summarize_diagnostics(profiles,segments,diagnostics)
        profiles,segments,diagnostics=fixture()
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            summarize_diagnostics(profiles+[profiles[0]],segments,diagnostics)


if __name__=="__main__":unittest.main(verbosity=2)
