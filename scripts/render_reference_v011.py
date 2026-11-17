"""Rebuild all six v011 figures from frozen plot inputs, without training.

An installed/licensed Times New Roman regular font is required for the archived
panel-label typography. Original figure files are preserved in reference/v011.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
from importlib.metadata import version

from verify_v011_reference import FIGURES, REFERENCE, ROOT, verify


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_font(override=None):
    from matplotlib.font_manager import FontProperties, findfont
    from matplotlib.ft2font import FT2Font
    requested = override or os.environ.get("TIMES_NEW_ROMAN_FONT")
    if requested:
        path = Path(requested).expanduser().resolve()
    elif Path("C:/Windows/Fonts/times.ttf").is_file():
        path = Path("C:/Windows/Fonts/times.ttf")
    else:
        path = Path(findfont(FontProperties(family="Times New Roman", style="normal", weight="normal"), fallback_to_default=False))
    if (not path.is_file() or FontProperties(fname=path).get_name() != "Times New Roman"
            or FT2Font(str(path)).style_flags != 0):
        raise ValueError("--font-path must identify an installed/licensed Times New Roman regular font file")
    os.environ["TIMES_NEW_ROMAN_FONT"] = str(path)
    return path


def schematic(output):
    """Draw the original schematic, then apply the v009/v011 grayscale palette."""
    import fitz
    import matplotlib.pyplot as plt
    from publication_v011 import builder
    palette = {"#0072b2": "#2e2e2e", "#555555": "#555555", "#c8d0d9": "#b8b8b8",
               "#d55e00": "#1a1a1a", "#eaf4fa": "#eeeeee", "#f2f5f8": "#f3f3f3",
               "#f7f8fa": "#f7f7f7", "#fdf1ea": "#dddddd", "#ffffff": "#ffffff"}
    def save(fig, stem):
        pdf, svg = output / (stem + ".pdf"), output / (stem + ".svg")
        fig.savefig(pdf, metadata={"Creator": "Rotor-temperature v011 schematic"})
        fig.savefig(svg)
        plt.close(fig)
        rgb = re.compile(rb"([-+\d.]+)\s+([-+\d.]+)\s+([-+\d.]+)\s+(rg|RG)\b")
        def recolor(match):
            color = "#" + "".join(f"{round(float(v)*255):02x}" for v in match.group(1, 2, 3))
            gray = int(palette[color][1:3], 16) / 255
            return f"{gray:.10f} ".encode() + (b"g" if match.group(4) == b"rg" else b"G")
        with fitz.open(pdf) as document:
            for xref in document[0].get_contents():
                before = document.xref_stream(xref)
                after = rgb.sub(recolor, before).replace(b"/DeviceRGB", b"/DeviceGray")
                def geometry(stream):
                    stream = rgb.sub(b"", stream)
                    stream = re.sub(rb"[-+\d.]+\s+[gG]\b", b"", stream)
                    return re.sub(rb"/Device(?:RGB|Gray)\s+[cC][sS]", b"", stream)
                assert geometry(before) == geometry(after)
                document.update_stream(xref, after)
            temp = pdf.with_suffix(".gray.pdf")
            document.save(temp, deflate=True)
        temp.replace(pdf)
        svg.write_text(re.sub(r"#[0-9a-fA-F]{6}", lambda m: palette[m.group().lower()], svg.read_text(encoding="utf-8")), encoding="utf-8")
        with fitz.open(pdf) as document:
            document[0].get_pixmap(dpi=300, colorspace=fitz.csGRAY).save(output / (stem + ".png"))
    builder.save = save
    builder.method()
    return {"status": "PASS", "programmatic_schematic": True,
            "color_conversion_preserves_all_non_color_pdf_operators": True, "palette": palette}


def inspect_figures(directory):
    import fitz
    import numpy as np
    from PIL import Image
    records = []
    for number, stem in FIGURES.items():
        with fitz.open(directory / (stem + ".pdf")) as current, fitz.open(REFERENCE / "figures" / (stem + ".pdf")) as saved:
            assert len(current) == 1 and not current[0].get_images(), stem
            assert current[0].mediabox == saved[0].mediabox and current[0].cropbox == saved[0].cropbox, stem
            assert current[0].get_text() == saved[0].get_text(), (stem, "text differs")
            a = current[0].get_pixmap(dpi=100, alpha=False)
            b = saved[0].get_pixmap(dpi=100, alpha=False)
            pixel_equal = a.samples == b.samples
        with Image.open(directory / (stem + ".png")) as bitmap:
            pixels = np.asarray(bitmap.convert("RGB"))
            gray = bool(np.array_equal(pixels[:, :, 0], pixels[:, :, 1]) and np.array_equal(pixels[:, :, 1], pixels[:, :, 2]))
            assert gray == (number in (1, 3, 5)), stem
        records.append({"paper_figure": number, "stem": stem, "grayscale": gray,
                        "vector_pdf": True, "archive_text_and_page_boxes_match": True,
                        "rendered_pdf_pixels_match_archive_at_100dpi": pixel_equal,
                        "sha256": {ext: digest(directory / (stem + "." + ext)) for ext in ("pdf", "svg", "png")}})
    return records


def render(output, font_path=None):
    if not __debug__:
        raise ValueError("Run without -O; drawing invariance assertions must be enabled")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError("Output already exists; select a new directory to preserve prior results")
    if output.is_relative_to(REFERENCE):
        raise ValueError("Output must be outside the immutable reference archive")
    check = verify()
    font = choose_font(font_path)
    import matplotlib.pyplot as plt
    from publication_v011 import summary, timeseries
    figures = output / "figures"
    review = output / "review"
    figures.mkdir(parents=True)
    review.mkdir()
    with plt.rc_context():
        scheme_receipt = schematic(figures)
    with plt.rc_context():
        summary.main(project_root=ROOT, output_directory=figures, source_pdf_directory=figures,
                     receipt_path=review / "summary-regeneration-receipt.json")
    data = REFERENCE / "data"
    curves = timeseries.read_csv(data / "figure5-plotted-data.csv")
    segments = timeseries.read_csv(data / "initialization-summary.csv")
    examples = json.loads((data / "selected-examples.json").read_text(encoding="utf-8"))
    time_receipts = {}
    for stem, make in ((FIGURES[4], lambda: timeseries.trajectories(curves, examples)),
                       (FIGURES[6], lambda: timeseries.segments(segments))):
        with plt.rc_context():
            plt.rcdefaults()
            fig, lines = make()
            time_receipts[stem] = timeseries.export(fig, lines, stem, output, None)
    (review / "timeseries-regeneration-receipt.json").write_text(json.dumps(time_receipts, indent=2) + "\n", encoding="utf-8")
    result = {"status": "PASS", "revision": "v011", "new_fits": 0, "new_predictions": 0,
              "source_verification": check, "schematic": scheme_receipt,
              "font": {"family": "Times New Roman", "sha256": digest(font), "redistributed": False},
              "runtime": {name: version(name) for name in ("numpy", "matplotlib", "Pillow", "PyMuPDF")},
              "figures": inspect_figures(figures),
              "scope": "Saved plot inputs; exact data/axes assertions, archived text/page boxes and print styles; file bytes depend on renderer metadata/environment."}
    verify()  # Inputs and archived assets must remain immutable through rendering.
    (review / "v011-regeneration-receipt.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "work/reference-v011")
    parser.add_argument("--font-path", type=Path)
    args = parser.parse_args()
    result = render(args.output, args.font_path)
    print(json.dumps({"status": result["status"], "revision": "v011", "figures": len(result["figures"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
