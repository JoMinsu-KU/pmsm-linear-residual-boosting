"""Fresh rendering checks: scientific values survive the v011 style changes."""
from contextlib import contextmanager
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pmsm_repro import plot_style
from pmsm_repro.reporting import _draw_figures


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


failure = load_script("analyze_failure")
initialization = load_script("analyze_initialization")


@contextmanager
def capture_figures():
    """Keep real saved artists available for numerical and layout inspection."""
    figures = {}
    original = matplotlib.figure.Figure.savefig
    def save(fig, path, *args, **kwargs):
        figures[Path(path).stem] = fig
        return original(fig, path, *args, **kwargs)
    with patch.object(matplotlib.figure.Figure, "savefig", save), patch.object(plt, "close"):
        yield figures
    plt.close("all")


class FreshV011PlotTests(unittest.TestCase):
    def assert_panels(self, fig, receipt, expected):
        labels = receipt["panel_labels"]
        self.assertEqual(len(labels), expected)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for entry in labels:
            text = next(t for t in fig.texts if t.get_text() == entry["text"])
            ax = fig.axes[entry["axis_index"]]
            bounds = text.get_window_extent(renderer).transformed(fig.transFigure.inverted())
            axis = ax.get_position()
            self.assertAlmostEqual((bounds.x0 + bounds.x1) / 2, (axis.x0 + axis.x1) / 2, places=6)
            self.assertLess(bounds.y1, axis.y0)
            self.assertGreater(bounds.y0, 0., entry["text"])
            self.assertAlmostEqual(text.get_fontsize() * entry["insertion_scale"], 8.)
            if ax.get_xlabel():
                xlabel = ax.xaxis.label.get_window_extent(renderer).transformed(fig.transFigure.inverted())
                self.assertLess(bounds.y1, xlabel.y0)

    def test_summary_values_markers_hatches_grayscale_and_panel_labels(self):
        labels = ["Selected direct boosting", "Linear residual boosting", "Equal-weight ensemble",
                  "Matched direct boosting", "Expanded direct boosting", "History-only residual boosting",
                  "History-only ensemble", "Neural network (seed 11)"]
        points = [dict(cohort="Development", method=label, median_us=10. + i, macro_rmse=2. + i)
                  for i, label in enumerate(labels)]
        paired = [dict(cohort="Development", profile="1", rmse_difference=-.5),
                  dict(cohort="Development", profile="2", rmse_difference=.3)]
        changes = [dict(dataset="IPMSM", cohort="Historical evaluation", method=label, change_percent=value)
                   for label, value in zip([labels[0], labels[1], labels[2], labels[6], "Neural network metric mean"],
                                           [-15., 3., -2., 5., -7.])]
        with tempfile.TemporaryDirectory() as tmp, capture_figures() as figures:
            out = Path(tmp); (out / "figures").mkdir()
            _draw_figures(out, points, paired, changes, "smoke")
            receipt = json.loads((out / "figures/plot-style-receipt.json").read_text())
            self.assertFalse(receipt["exact_archive_layout"])
            self.assertEqual(len(receipt["figures"]), 4)
            for stem in ("figure1-method-overview", "figure3-paired-profile-errors", "figure4-initialization-effect"):
                pixels = plt.imread(out / "figures" / (stem + ".png"))
                np.testing.assert_array_equal(pixels[..., 0], pixels[..., 1])
                np.testing.assert_array_equal(pixels[..., 1], pixels[..., 2])
            scatter = figures["figure2-accuracy-latency"].axes[0].collections
            for i, (artist, marker) in enumerate(zip(scatter, ["^", "o", "D", "s", "v", "P", "h", "*"])):
                np.testing.assert_allclose(artist.get_offsets(), [[10. + i, 2. + i]])
                shape = MarkerStyle(marker)
                np.testing.assert_array_equal(artist.get_paths()[0].vertices,
                                              shape.get_path().transformed(shape.get_transform()).vertices)
            for i in (0, 3, 6):
                self.assertEqual(len(scatter[i].get_facecolors()), 0)
            bars = figures["figure3-paired-profile-errors"].axes[0].patches
            self.assertEqual([b.get_height() for b in bars], [-.5, .3])
            self.assertEqual([b.get_hatch() for b in bars], ["/", "xx"])
            bars = figures["figure4-initialization-effect"].axes[0].patches
            self.assertEqual([b.get_width() for b in bars], [-15., 3., -2., 5., -7.])
            self.assertEqual([b.get_hatch() for b in bars], [None, "///", "\\\\\\", "xx", ".."])
            effect = figures["figure4-initialization-effect"]
            effect.canvas.draw()
            renderer = effect.canvas.get_renderer()
            for text in effect.axes[0].texts:
                bounds = text.get_window_extent(renderer)
                self.assertGreaterEqual(bounds.x0, effect.axes[0].get_window_extent(renderer).x0)
                self.assertLessEqual(bounds.x1, effect.bbox.x1)
            for stem, count in [("figure2-accuracy-latency", 2), ("figure3-paired-profile-errors", 2),
                                ("figure4-initialization-effect", 3)]:
                self.assert_panels(figures[stem], receipt["figures"][stem], count)
                self.assertTrue(any(t.get_text() == "smoke" for t in figures[stem].texts))
            self.assertEqual(receipt["figures"]["figure4-initialization-effect"]["paper_figure"], 5)

    def test_marker_assignment_survives_missing_earlier_methods(self):
        with tempfile.TemporaryDirectory() as tmp, capture_figures() as figures:
            out = Path(tmp); (out / "figures").mkdir()
            _draw_figures(out, [dict(cohort="Historical evaluation", method="Matched direct boosting",
                                    median_us=12., macro_rmse=1.)], [], [], "smoke")
            fig = figures["figure2-accuracy-latency"]
            artist = fig.axes[1].collections[0]
            shape = MarkerStyle("s")
            np.testing.assert_array_equal(artist.get_paths()[0].vertices,
                                          shape.get_path().transformed(shape.get_transform()).vertices)
            self.assertEqual([t.get_text() for t in fig.legends[0].get_texts()], ["Matched direct boosting"])

    def test_single_trajectory_keeps_every_endpoint_and_smoke_label(self):
        index = np.arange(300, 342)
        target = 20. + np.sin(np.arange(len(index)) / 5.)
        arrays = {"index": index, **{"prediction__" + method: target + shift
                  for method, shift in [("linear_prior", 2.), ("linear_residual", .5), ("direct200", 1.)]}}
        record = dict(dataset="paderborn", arm="own_boundary", role="confirmation", profile=1,
                      linear_prior_rmse_c=2., linear_residual_rmse_c=.5, direct200_rmse_c=1.)
        with tempfile.TemporaryDirectory() as tmp, capture_figures() as figures:
            out = Path(tmp)
            failure.plot_examples([dict(arrays=arrays, target=target, metrics=record)], out, smoke=True)
            stem = "figure5-temperature-trajectories"
            fig = figures[stem]
            for line, method in zip(fig.axes[0].lines, ["target", "linear_prior", "linear_residual", "direct200"]):
                np.testing.assert_array_equal(line.get_xdata(), index)
                np.testing.assert_array_equal(line.get_ydata(), target if method == "target" else arrays["prediction__" + method])
            prior, residual, matched = fig.axes[0].lines[1:]
            self.assertEqual([p.get_marker() for p in [prior, residual, matched]], ["D", "o", "s"])
            self.assertEqual([prior.get_markerfacecolor(), matched.get_markerfacecolor()], ["white", "white"])
            self.assertEqual(residual.get_markerfacecolor(), "#0072B2")
            self.assertEqual(matched.get_linestyle(), "--")
            self.assertTrue(residual.is_dashed())
            self.assertNotEqual(residual._unscaled_dash_pattern, matched._unscaled_dash_pattern)
            receipt = json.loads((out / "plot-style-receipt.json").read_text())["figures"][stem]
            self.assertEqual(receipt["paper_figure"], 4)
            self.assert_panels(fig, receipt, 2)
            self.assertTrue(any("SMOKE TEST" in t.get_text() for t in fig.texts))
            with (out / "figure5-plotted-data.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            np.testing.assert_array_equal([float(r["target_c"]) for r in rows], target)
            np.testing.assert_array_equal([int(r["within_profile_row"]) for r in rows], index)

    def test_segment_subset_preserves_values_and_marks_missing_panels(self):
        rows = [dict(dataset="ipmsm", role="pilot", method=method, fifth=fifth,
                     macro_difference=shift * fifth, profiles=2)
                for method, shift in [("linear_residual", -.2), ("direct200", -.1), ("d9", .1)]
                for fifth in range(1, 6)]
        with tempfile.TemporaryDirectory() as tmp, capture_figures() as figures:
            out = Path(tmp); (out / "figures").mkdir()
            initialization.plot(rows, out, smoke=True)
            stem = "figure6-initialization-segments"
            fig = figures[stem]
            for line, shift, marker in zip(fig.axes[1].lines, [-.2, -.1, .1], ["o", "s", "^"]):
                np.testing.assert_array_equal(line.get_xdata(), range(1, 6))
                np.testing.assert_array_equal(line.get_ydata(), [shift * i for i in range(1, 6)])
                self.assertEqual(line.get_marker(), marker)
            self.assertEqual(fig.axes[1].lines[2].get_markerfacecolor(), "white")
            self.assertEqual(fig.axes[1].lines[2].get_color(), "#009E73")
            self.assertTrue(any("Not available" in t.get_text() for t in fig.axes[0].texts))
            receipt = json.loads((out / "figures/plot-style-receipt.json").read_text())["figures"][stem]
            self.assert_panels(fig, receipt, 5)
            self.assertTrue(any("SMOKE TEST" in t.get_text() for t in fig.texts))
            renderer = fig.canvas.get_renderer()
            legend = fig.axes[5].get_legend().get_window_extent(renderer)
            note = next(t for t in fig.texts if t.get_text().startswith("Change:")).get_window_extent(renderer)
            self.assertLess(note.y1, legend.y0)
            self.assertLessEqual(legend.x1, fig.bbox.x1)

    def test_font_substitute_is_explicit_and_keeps_final_eight_point_size(self):
        with patch.object(plot_style, "_times_font_path", return_value=None):
            fig, ax = plt.subplots(figsize=(7.2, 3.0))
            fig.subplots_adjust(bottom=.3)
            receipt = plot_style.add_panel_labels(fig, [ax], 177.53)
            self.assertEqual(receipt["font_family"], "DejaVu Serif")
            self.assertTrue(receipt["font_fallback_used"])
            self.assertFalse(receipt["times_new_roman"])
            self.assertFalse(receipt["font_redistributed"])
            self.assertIn("not exact", receipt["font_status"])
            self.assert_panels(fig, receipt, 1)
            plt.close(fig)

    def test_explicit_font_override_cannot_silently_fall_back(self):
        with patch.dict("os.environ", {"PMSM_TIMES_FONT": "missing-local-font.ttf"}):
            with self.assertRaisesRegex(FileNotFoundError, "PMSM_TIMES_FONT"):
                plot_style.panel_font()
        properties, receipt = plot_style.panel_font()
        self.assertEqual(properties.get_name(), receipt["font_family"])
        if receipt["times_new_roman"]:
            self.assertEqual(receipt["font_family"], "Times New Roman")

    def test_explicit_bold_or_italic_times_is_rejected(self):
        from matplotlib import font_manager
        from matplotlib.ft2font import FT2Font
        candidates = []
        for options in [dict(weight="bold"), dict(style="italic")]:
            try:
                path = font_manager.findfont(font_manager.FontProperties(family="Times New Roman", **options),
                                             fallback_to_default=False)
            except ValueError:
                continue
            if FT2Font(path).style_flags:
                candidates.append(path)
        if not candidates:
            self.skipTest("No local non-regular Times New Roman available")
        for path in candidates:
            with self.subTest(path=path), patch.dict("os.environ", {"PMSM_TIMES_FONT": path}):
                with self.assertRaisesRegex(ValueError, "regular Times New Roman"):
                    plot_style.panel_font()


if __name__ == "__main__":
    unittest.main(verbosity=2)
