"""Restyle saved-result Figures 2, 3 and 5; no training, inference or refitting.

Run from any directory: python replot/summary/replot_summary_figures.py
The adapted historical generator and exact saved inputs accompany this module.  Each original plot is captured and compared to its
restyled counterpart before export. Figure 2 retains its original colors plus
eight distinct marker shapes; Figures 3 and 5 use gray fills and hatches.
Figure 5 retains its historical asset
stem, figure4-initialization-effect.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import re
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.markers import MarkerStyle
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from PIL import Image
import fitz

BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parents[1]
GENERATOR = BUNDLE / "builder.py"
OUT = ROOT / "work/reference-v011/figures"
SOURCE_OUT = OUT
REVIEW = ROOT / "work/reference-v011/review/summary-regeneration-receipt.json"
EXPECTED_BOXES = ROOT / "reference/v011/expected_pdf_boxes.json"
PANEL_PREFIX = re.compile(r"^\([a-z]\)\s+")

METHODS = {
    "Selected direct boosting": ("^", "#555555", "none"),
    "Linear residual boosting": ("o", "#0072B2", "#0072B2"),
    "Equal-weight ensemble": ("D", "#D55E00", "#D55E00"),
    "Matched direct boosting (200 trees)": ("s", "#555555", "none"),
    "Expanded direct boosting": ("v", "#555555", "#555555"),
    "History-only residual boosting": ("P", "#CC79A7", "#CC79A7"),
    "History-only ensemble": ("h", "#CC79A7", "none"),
    "Neural network (timed seed)": ("*", "#009E73", "#009E73"),
}
FIGURE_NUMBERS = {"figure2-accuracy-latency": 2,
                  "figure3-paired-profile-errors": 3,
                  "figure4-initialization-effect": 5}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(path):
    path = Path(path).resolve()
    # Windows output folders can be on another drive; relpath cannot span drives.
    try:
        display = Path(os.path.relpath(path, ROOT)).as_posix()
    except ValueError:
        display = path.as_posix()
    return {"path": display,
            "sha256": sha(path), "bytes": Path(path).stat().st_size}


def geometry(fig):
    """Plot data and geometry, excluding explicitly editable paint/marker styles."""
    axes = []
    for ax in fig.axes:
        axes.append({
            "xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()),
            "xscale": ax.get_xscale(), "yscale": ax.get_yscale(),
            "position": list(ax.get_position().bounds),
            "xticks": ax.get_xticks().tolist(), "yticks": ax.get_yticks().tolist(),
            "xticklabels": [x.get_text() for x in ax.get_xticklabels()],
            "yticklabels": [x.get_text() for x in ax.get_yticklabels()],
            "scatter_offsets": [np.asarray(x.get_offsets()).tolist()
                                for x in ax.collections if isinstance(x, PathCollection)],
            "scatter_sizes": [x.get_sizes().tolist() for x in ax.collections
                              if isinstance(x, PathCollection)],
            "rectangles": [list(x.get_bbox().bounds) for x in ax.patches
                           if isinstance(x, Rectangle)],
            "lines": [{"x": np.asarray(x.get_xdata()).tolist(),
                       "y": np.asarray(x.get_ydata()).tolist()} for x in ax.lines],
            "texts": [{"text": x.get_text(), "position": list(x.get_position())}
                      for x in ax.texts],
            "title": ax.get_title(loc="left"),
            "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
        })
    return json.loads(json.dumps(
        {"size_inches": fig.get_size_inches().tolist(), "axes": axes},
        default=lambda value: value.item()))


def style_text_and_axes(fig):
    for text in fig.findobj(Text):
        text.set_color("#000000")
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_color("#000000")
        ax.tick_params(colors="black")
        for grid in ax.get_xgridlines() + ax.get_ygridlines():
            grid.set_color("#808080")


def style_scatter(fig):
    for ax in fig.axes:
        for collection in ax.collections:
            marker, edge, face = METHODS[collection.get_label()]
            shape = MarkerStyle(marker)
            collection.set_paths([shape.get_path().transformed(shape.get_transform())])
            collection.set_edgecolor(edge)
            collection.set_facecolor(face)
            collection.set_linewidth(.85)
    # Legend is regenerated from the restyled artists at the original anchor.
    for legend in list(fig.legends):
        legend.remove()
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.53, .025),
               ncol=2, frameon=False, columnspacing=2.1, handletextpad=.6,
               labelspacing=.65, fontsize=8)


def style_paired(fig):
    for ax in fig.axes:
        for bar in ax.patches:
            negative = bar.get_height() < 0
            bar.set_facecolor("#bdbdbd" if negative else "#ffffff")
            bar.set_edgecolor("#000000")
            bar.set_linewidth(.65)
            bar.set_hatch("/" if negative else "xx")
    # Two compact lines fit the original footer band without shrinking type.
    note = fig.texts[0]
    note.set_text("Negative /: favors linear residual.\nPositive x: favors matched direct.")
    note.set_position((.5, .022))
    note.set_linespacing(1.25)


def style_initialization(fig):
    hatches = [None, "///", "\\\\\\", "xx", ".."]
    fills = ["#aaaaaa", "#ffffff", "#e5e5e5", "#ffffff", "#eeeeee"]
    for ax in fig.axes:
        for bar, hatch, fill in zip(ax.patches, hatches, fills):
            bar.set_facecolor(fill)
            bar.set_edgecolor("#000000")
            bar.set_linewidth(.65)
            bar.set_hatch(hatch)
        for text in ax.texts:
            # Only long negative labels lie inside their bars in the original.
            try:
                value = float(text.get_text())
            except ValueError:
                continue
            if value < -25:
                # Move the three long negative labels inward so their white
                # patches do not obscure the bars' left endpoint outlines.
                x, y = text.get_position()
                text.set_position((x + 3.0, y))
                text.set_color("#000000")
                text.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": .6})


def times_new_roman_path():
    candidates = []
    if os.environ.get("TIMES_NEW_ROMAN_FONT"):
        candidates.append(Path(os.environ["TIMES_NEW_ROMAN_FONT"]))
    candidates.append(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/times.ttf")
    try:
        candidates.append(Path(font_manager.findfont(
            font_manager.FontProperties(family="Times New Roman"), fallback_to_default=False)))
    except ValueError:
        pass
    for path in candidates:
        if path.is_file():
            properties = font_manager.FontProperties(fname=str(path))
            if properties.get_name() == "Times New Roman":
                return path
    raise RuntimeError("Times New Roman is required for template panel labels. Install it locally or set TIMES_NEW_ROMAN_FONT to a licensed times.ttf file; the font is not redistributed with this package.")


def add_template_panel_labels(fig, stem, font_path):
    # Preserve every axis box. Only title prefixes, added panel letters, and
    # the two crowded footer placements are changed for template compliance.
    for ax in fig.axes:
        ax._left_title.set_text(PANEL_PREFIX.sub("", ax.get_title(loc="left")))
        for text in ax.texts:
            text.set_text(PANEL_PREFIX.sub("", text.get_text()))
    if stem == "figure3-paired-profile-errors":
        fig.texts[0].set_position((.5, .007))
        fig.texts[0].set_linespacing(1.0)
    elif stem == "figure4-initialization-effect":
        fig.texts[0].set_position((.5, .009))
    target_width_mm = 177.53 if stem == "figure2-accuracy-latency" else 85.29
    width_pt, height_pt = fig.get_size_inches() * 72
    scale = target_width_mm / 25.4 * 72 / width_pt
    native_size = 8.0 / scale
    properties = font_manager.FontProperties(family="Times New Roman", fname=str(font_path), size=native_size)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = []
    for index, ax in enumerate(fig.axes):
        box = ax.get_position()
        if ax.get_xlabel():
            label_box = ax.xaxis.label.get_window_extent(renderer).transformed(fig.transFigure.inverted())
            top = label_box.y0 - 2.0 / height_pt
        else:
            top = box.y0 - 6.0 / height_pt
        text = fig.text((box.x0 + box.x1) / 2, top, f"({chr(97 + index)})",
                        ha="center", va="top", fontproperties=properties, color="black")
        labels.append({"text": text.get_text(), "position": list(text.get_position()),
                       "native_font_points": native_size, "final_font_points": native_size * scale,
                       "final_include_width_mm": target_width_mm, "font": "Times New Roman"})
    return labels


def convert_pdf_to_devicegray(path):
    """Convert only equal-RGB painting operators; preserve all geometry/fonts."""
    number = rb"[-+]?(?:\d*\.\d+|\d+\.?)(?:[Ee][-+]?\d+)?"
    triple = re.compile(rb"(?<![\w.])(" + number + rb")\s+(" + number + rb")\s+(" + number + rb")\s+(rg|RG)\b")
    single = re.compile(rb"(?<![\w.])" + number + rb"\s+[gG]\b")
    colorspace = re.compile(rb"/Device(?:RGB|Gray)\s+[cC][sS]\b")
    def no_paint(stream):
        stream = triple.sub(b"", stream)
        stream = single.sub(b"", stream)
        return colorspace.sub(b"", stream).split()
    def replacement(match):
        values = [float(match.group(i)) for i in (1, 2, 3)]
        assert values[0] == values[1] == values[2], "Cannot convert a non-gray paint color"
        return match.group(1) + (b" g" if match.group(4) == b"rg" else b" G")
    with fitz.open(path) as document:
        before_fonts = document[0].get_fonts(full=True)
        before_pixels = document[0].get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False).samples
        streams = set(document[0].get_contents())
        streams.update(xref for xref in range(1, document.xref_length())
                       if document.xref_is_stream(xref) and "/PatternType" in document.xref_object(xref))
        for xref in streams:
            before = document.xref_stream(xref)
            after = triple.sub(replacement, before).replace(b"/DeviceRGB", b"/DeviceGray")
            assert no_paint(before) == no_paint(after), "A non-color PDF operator changed"
            assert b"/DeviceRGB" not in after and not triple.search(after)
            document.update_stream(xref, after)
        temporary = path.with_suffix(".devicegray.pdf")
        document.save(temporary, garbage=0, deflate=True)
    with fitz.open(temporary) as document:
        assert document[0].get_fonts(full=True) == before_fonts
        after_pixels = document[0].get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False).samples
        pixel_difference = np.abs(np.frombuffer(before_pixels, dtype=np.uint8).astype(int)
                                  - np.frombuffer(after_pixels, dtype=np.uint8).astype(int))
        assert int(pixel_difference.max()) <= 1
        assert all("/DeviceRGB" not in document.xref_object(xref)
                   for xref in range(1, document.xref_length()))
    temporary.replace(path)
    return {"colorspace": "DeviceGray", "color_only_stream_rewrite": True,
            "paths_and_fonts_unchanged": True, "converted_streams": len(streams),
            "maximum_render_channel_difference": int(pixel_difference.max())}


def check_pdf(path, original_boxes, labels):
    with fitz.open(path) as document:
        page = document[0]
        fonts = page.get_fonts(full=True)
        boxes = {"mediabox": list(page.mediabox), "cropbox": list(page.cropbox)}
        if original_boxes is not None:
            assert boxes == original_boxes, (path, boxes, original_boxes)
        assert not page.get_images(full=True), "Unexpected raster content in vector PDF"
        assert fonts and all("Type3" not in str(font) for font in fonts)
        panel_spans = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    color = span["color"]
                    assert (color >> 16 & 255) == (color >> 8 & 255) == (color & 255)
                    if re.fullmatch(r"\([a-z]\)", span["text"]):
                        expected = next(label for label in labels if label["text"] == span["text"])
                        assert "TimesNewRoman" in span["font"]
                        assert abs(span["size"] - expected["native_font_points"]) < 1e-4
                        panel_spans.append({"text": span["text"], "font": span["font"],
                                            "size": span["size"], "bbox": span["bbox"]})
        assert len(panel_spans) == len(labels)
        return {"boxes": boxes, "embedded_fonts": [list(font) for font in fonts],
                "no_raster_images": True, "no_type3_fonts": True,
                "text_is_grayscale": True, "panel_label_spans": panel_spans}


def main(project_root=None, output_directory=None, source_pdf_directory=None,
         receipt_path=None, launcher=None):
    global ROOT, OUT, SOURCE_OUT, REVIEW
    if project_root is not None:
        ROOT = Path(project_root)
    if output_directory is not None:
        OUT = Path(output_directory)
    if source_pdf_directory is not None:
        SOURCE_OUT = Path(source_pdf_directory)
    if receipt_path is not None:
        REVIEW = Path(receipt_path)
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    expected_boxes = json.loads(EXPECTED_BOXES.read_text(encoding="utf-8"))
    panel_font_path = times_new_roman_path()
    spec = importlib.util.spec_from_file_location("saved_publication_figures", GENERATOR)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "svg.hashsalt": "c18-access-v011-template",
                         "hatch.linewidth": .55})
    records = []

    def save_styled(fig, stem):
        fig.canvas.draw()
        before = geometry(fig)
        original_boxes = expected_boxes[stem]
        if stem == "figure2-accuracy-latency":
            style_scatter(fig)
        elif stem == "figure3-paired-profile-errors":
            style_paired(fig)
        elif stem == "figure4-initialization-effect":
            style_initialization(fig)
        else:
            raise ValueError(stem)
        panel_labels = add_template_panel_labels(fig, stem, panel_font_path)
        style_text_and_axes(fig)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for text, label in zip(fig.texts[-len(panel_labels):], panel_labels):
            box = text.get_window_extent(renderer)
            assert box.x0 > 0 and box.y0 > 0 and box.x1 < fig.bbox.width and box.y1 < fig.bbox.height
            for other in fig.findobj(Text):
                if other is not text and other.get_visible() and other.get_text():
                    assert not box.overlaps(other.get_window_extent(renderer)), (stem, text.get_text(), other.get_text())
            label["native_bbox_inches"] = (np.asarray(box.bounds) / fig.dpi).tolist()
        after = geometry(fig)
        annotation_exceptions = []
        comparison = json.loads(json.dumps(after))
        for old_axis, new_axis, compared_axis in zip(before["axes"], after["axes"], comparison["axes"]):
            assert new_axis["title"] == PANEL_PREFIX.sub("", old_axis["title"])
            compared_axis["title"] = old_axis["title"]
            for old_text, new_text, compared_text in zip(old_axis["texts"], new_axis["texts"], compared_axis["texts"]):
                assert new_text["text"] == PANEL_PREFIX.sub("", old_text["text"])
                compared_text["text"] = old_text["text"]
        if stem == "figure4-initialization-effect":
            for panel, (old_axis, new_axis) in enumerate(zip(before["axes"], after["axes"])):
                for index, (old_text, new_text) in enumerate(zip(old_axis["texts"], new_axis["texts"])):
                    if old_text["position"] == new_text["position"]:
                        continue
                    assert float(old_text["text"]) < -25
                    assert new_text["position"] == [old_text["position"][0] + 3.0, old_text["position"][1]]
                    assert new_text["text"] == old_text["text"]
                    annotation_exceptions.append({"panel": panel + 1,
                                                  "text": old_text["text"],
                                                  "original_position": old_text["position"],
                                                  "restyled_position": new_text["position"],
                                                  "reason": "Move white-backed label 3.0 x units inside the bar to expose its endpoint outline."})
                    comparison["axes"][panel]["texts"][index]["position"] = old_text["position"]
            assert len(annotation_exceptions) == 3
        assert before == comparison, f"Scientific data or plot geometry changed: {stem}"
        baseline_sha = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
        for extension in ("pdf", "svg", "png"):
            fig.savefig(OUT / f"{stem}.{extension}", dpi=300, facecolor="white",
                        metadata={"Creator": "Rotor-temperature v011 saved figures"})
        devicegray = None
        if stem != "figure2-accuracy-latency":
            devicegray = convert_pdf_to_devicegray(OUT / f"{stem}.pdf")
            with Image.open(OUT / f"{stem}.png") as bitmap:
                bitmap.convert("L").save(OUT / f"{stem}.png", dpi=(300, 300))
        else:
            with Image.open(OUT / f"{stem}.png") as bitmap:
                bitmap.convert("RGB").save(OUT / f"{stem}.png", dpi=(300, 300))
        if SOURCE_OUT.resolve() != OUT.resolve():
            shutil.copy2(OUT / f"{stem}.pdf", SOURCE_OUT / f"{stem}.pdf")
        rgb = np.asarray(Image.open(OUT / f"{stem}.png").convert("RGB"))
        grayscale = (np.array_equal(rgb[..., 0], rgb[..., 1])
                     and np.array_equal(rgb[..., 1], rgb[..., 2]))
        if stem != "figure2-accuracy-latency":
            assert grayscale
        else:
            assert not grayscale, "Figure 2 must retain its original method colors"
        records.append({"stem": stem, "paper_figure": FIGURE_NUMBERS[stem],
                        "size_inches": before["size_inches"], "dpi": 300,
                        "original_and_annotation_normalized_geometry_sha256": baseline_sha,
                        "plotted_arrays_axes_ranges_order_and_geometry_exact": True,
                        "annotation_position_exceptions": annotation_exceptions,
                        "template_panel_labels": panel_labels,
                        "template_text_changes": "Removed title letter prefixes; added centered lower Times New Roman panel labels, final size 8 pt; moved footer notes only where needed.",
                        "geometry": after,
                        "rendering": "original colors plus distinct markers" if stem == "figure2-accuracy-latency" else "grayscale and hatches",
                        "png_channels_identical": grayscale,
                        "png_mode": Image.open(OUT / f"{stem}.png").mode,
                        "devicegray_conversion": devicegray,
                        "pdf_checks": check_pdf(OUT / f"{stem}.pdf", original_boxes, panel_labels),
                        "files": [reference(OUT / f"{stem}.{ext}")
                                  for ext in ("pdf", "svg", "png")],
                        "source_pdf": reference(SOURCE_OUT / f"{stem}.pdf")})
        plt.close(fig)

    builder.save = save_styled
    builder.accuracy(builder.read(builder.SOURCES[0]), builder.read(builder.SOURCES[1]))
    builder.paired(builder.read(builder.SOURCES[0]))
    builder.initialization(builder.read(builder.SOURCES[2]))
    assert all(sha(path) == expected for path, expected in builder.SOURCE_HASHES.items())
    receipt = {"schema": "c18-v011-template-summary-v1", "status": "GENERATED",
               "scope": "Presentation only; saved metrics; no fitting, inference or new statistical tests. Figure 2 keeps original colors plus distinct markers; Figures 3 and 5 use grayscale and hatches.",
               "adapted_saved_generator": reference(GENERATOR),
               "wrapper": reference(Path(__file__)),
               "launcher": reference(launcher) if launcher else None,
               "expected_pdf_boxes": reference(EXPECTED_BOXES),
               "sources": [reference(path) for path in builder.SOURCES],
               "all_source_hashes_unchanged": True,
               "external_source_dependencies": [],
               "required_local_panel_font": {"name": "Times New Roman", "path": str(panel_font_path), "redistributed": False},
               "method_marker_mapping": METHODS,
               "paired_bar_encoding": {"negative": "midgray, / hatch, black edge",
                                       "positive": "white, xx cross hatch, black edge"},
               "initialization_row_hatches": ["plain", "///", "\\\\\\", "xx", ".."],
               "figures": records, "plotted_values": builder.NUMERIC}
    REVIEW.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "figures": len(records),
                      "receipt": str(REVIEW)}))


if __name__ == "__main__":
    main()
