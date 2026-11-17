"""Archive-integrity and preflight tests; mutate only temporary archive copies.

These tests need only the standard library except the explicitly skipped font
checks. They do not redraw figures, train models, or refresh repository manifests.
"""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


integrity = load_script("verify_v011_reference")
with patch.dict(sys.modules, {"verify_v011_reference": integrity}):
    renderer = load_script("render_reference_v011")


class ReferenceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / "reference"
        shutil.copytree(integrity.REFERENCE, self.reference)

    def read_manifest(self):
        return json.loads((self.reference / "manifest.json").read_text(encoding="utf-8"))

    def write_manifest(self, manifest):
        # This is a deliberately untrusted fixture, never the repository archive.
        (self.reference / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def refresh_fixture_entry(self, relative):
        manifest = self.read_manifest()
        entry = next(item for item in manifest["files"] if item["path"] == relative)
        content = (self.reference / relative).read_bytes()
        entry.update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        self.write_manifest(manifest)

    def change_csv_value(self, relative, column, amount):
        path = self.reference / relative
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        rows[0][column] = str(float(rows[0][column]) + amount)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        self.refresh_fixture_entry(relative)

    def test_intact_relocated_archive_passes(self):
        result = integrity.verify(self.reference)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["archived_figures"], 6)
        self.assertEqual(result["trajectory_rows"], 15339)
        self.assertEqual(result["initialization_cells"], 75)

    def test_same_length_tampering_is_rejected_by_hash(self):
        path = self.reference / "figures/figure1-method-overview.pdf"
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "Reference hash/size mismatch"):
            integrity.verify(self.reference)

    def test_duplicate_manifest_path_is_rejected(self):
        manifest = self.read_manifest()
        manifest["files"].append(dict(manifest["files"][0]))
        self.write_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "Duplicate reference manifest path"):
            integrity.verify(self.reference)

    def test_parent_and_absolute_path_escapes_are_rejected(self):
        outside = self.root / "outside.bin"
        outside.write_bytes(b"outside the reference root")
        baseline = self.read_manifest()
        for path in ("../outside.bin", str(outside.resolve())):
            with self.subTest(path=path):
                manifest = json.loads(json.dumps(baseline))
                manifest["files"].append({"path": path, "bytes": outside.stat().st_size,
                                          "sha256": hashlib.sha256(outside.read_bytes()).hexdigest()})
                self.write_manifest(manifest)
                with self.assertRaisesRegex(ValueError, "Unsafe reference path"):
                    integrity.verify(self.reference)
                self.assertEqual(outside.read_bytes(), b"outside the reference root")

    def test_unlisted_file_is_rejected(self):
        (self.reference / "data/unlisted.csv").write_text("unexpected\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "file inventory differs"):
            integrity.verify(self.reference)

    def test_trajectory_tamper_fails_even_after_fixture_hash_refresh(self):
        self.change_csv_value("data/figure5-plotted-data.csv", "linear_residual_c", 100.)
        with self.assertRaisesRegex(ValueError, "do not reproduce selected-example metrics"):
            integrity.verify(self.reference)

    def test_segment_tamper_fails_even_after_fixture_hash_refresh(self):
        self.change_csv_value("data/initialization-summary.csv", "macro_difference", 1.)
        with self.assertRaisesRegex(ValueError, "Initialization difference mismatch"):
            integrity.verify(self.reference)


class RenderPreflightTests(unittest.TestCase):
    def test_existing_directory_and_file_are_preserved_before_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "existing"
            directory.mkdir()
            sentinel = directory / "keep.txt"
            sentinel.write_bytes(b"do not replace")
            for output in (directory, sentinel):
                with self.subTest(output=output), patch.object(renderer, "verify") as verify:
                    with self.assertRaisesRegex(ValueError, "Output already exists"):
                        renderer.render(output)
                    verify.assert_not_called()
                    self.assertEqual(sentinel.read_bytes(), b"do not replace")
                    self.assertEqual(list(directory.iterdir()), [sentinel])

    def test_new_output_inside_reference_is_refused_without_creating_it(self):
        with tempfile.TemporaryDirectory() as temp:
            reference = Path(temp) / "reference"
            reference.mkdir()
            output = reference / "generated"
            with patch.object(renderer, "REFERENCE", reference), patch.object(renderer, "verify") as verify:
                with self.assertRaisesRegex(ValueError, "immutable reference archive"):
                    renderer.render(output)
                verify.assert_not_called()
            self.assertFalse(output.exists())

    def test_integrity_failure_creates_no_output_and_never_loads_font(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "new-output"
            with patch.object(renderer, "verify", side_effect=ValueError("corrupt reference")), \
                    patch.object(renderer, "choose_font") as font:
                with self.assertRaisesRegex(ValueError, "corrupt reference"):
                    renderer.render(output)
                font.assert_not_called()
            self.assertFalse(output.exists())


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Optional matplotlib is not installed")
class FontPreflightTests(unittest.TestCase):
    def assert_invalid_font_has_no_output(self, font):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "new-output"
            with patch.dict(os.environ), patch.object(renderer, "verify", return_value={"status": "PASS"}):
                with self.assertRaises((ValueError, RuntimeError, OSError)):
                    renderer.render(output, font)
            self.assertFalse(output.exists())

    def test_missing_font_is_refused_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assert_invalid_font_has_no_output(Path(temp) / "missing.ttf")

    def test_corrupt_font_is_refused_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            font = Path(temp) / "invalid.ttf"
            font.write_bytes(b"This is not a font.")
            self.assert_invalid_font_has_no_output(font)

    def test_other_font_family_is_refused_before_output_creation(self):
        from matplotlib.font_manager import FontProperties, findfont
        font = Path(findfont(FontProperties(family="DejaVu Sans"), fallback_to_default=False))
        self.assert_invalid_font_has_no_output(font)

    def test_times_bold_or_italic_is_refused_before_output_creation(self):
        from matplotlib import font_manager
        from matplotlib.ft2font import FT2Font
        variants = []
        for entry in font_manager.fontManager.ttflist:
            if entry.name == "Times New Roman":
                path = Path(entry.fname)
                if path.is_file() and FT2Font(str(path)).style_flags:
                    variants.append(path)
        if not variants:
            self.skipTest("No installed Times New Roman bold/italic face")
        for font in sorted(set(variants)):
            with self.subTest(font=font.name):
                self.assert_invalid_font_has_no_output(font)


if __name__ == "__main__":
    unittest.main()
