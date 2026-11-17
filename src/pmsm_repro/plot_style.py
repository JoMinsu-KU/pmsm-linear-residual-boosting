"""v011 visual conventions for fresh results, without archived data or fonts.

Fresh figures share the publication's method encodings. They do not claim the
archived page geometry. Panel letters are 8 pt at the stated insertion width;
the receipt explicitly distinguishes locally installed Times New Roman from
the portable DejaVu Serif substitute. No operating-system font is distributed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


SCATTER_STYLES = {
    "Selected direct boosting": ("^", "#555555", "none"),
    "Linear residual boosting": ("o", "#0072B2", "#0072B2"),
    "Equal-weight ensemble": ("D", "#D55E00", "#D55E00"),
    "Matched direct boosting": ("s", "#555555", "none"),
    "Expanded direct boosting": ("v", "#555555", "#555555"),
    "History-only residual boosting": ("P", "#CC79A7", "#CC79A7"),
    "History-only ensemble": ("h", "#CC79A7", "none"),
    "Neural network (timed seed)": ("*", "#009E73", "#009E73"),
}
LINE_STYLES = {
    "target": dict(color="#222222", linestyle="-", marker=""),
    "linear_prior": dict(color="#777777", linestyle=":", marker="D", markerfacecolor="white"),
    "linear_residual": dict(color="#0072B2", linestyle=(0, (3, 1.1, .8, 1.1)), marker="o", markerfacecolor="#0072B2"),
    "direct200": dict(color="#D55E00", linestyle="--", marker="s", markerfacecolor="white"),
    "d9": dict(color="#009E73", linestyle=":", marker="^", markerfacecolor="white"),
}
INITIALIZATION_BARS = (
    ("#aaaaaa", None), ("#ffffff", "///"), ("#e5e5e5", "\\\\\\"),
    ("#ffffff", "xx"), ("#eeeeee", ".."),
)
PAPER_FIGURE_NUMBERS = {
    "figure1-method-overview": 1, "figure2-accuracy-latency": 2,
    "figure3-paired-profile-errors": 3, "figure5-temperature-trajectories": 4,
    "figure4-initialization-effect": 5, "figure6-initialization-segments": 6,
}


def scatter_style(label):
    """A missing method never shifts another method's marker assignment."""
    key = "Neural network (timed seed)" if label.startswith("Neural network (seed ") else label
    marker, edge, face = SCATTER_STYLES[key]
    return dict(marker=marker, edgecolors=edge, facecolors=face, linewidths=.85)


def line_style(method, count=None):
    result = dict(LINE_STYLES[method])
    if method != "target":
        result.update(markeredgecolor=result["color"], markeredgewidth=.75)
        if count is not None:
            step = max(1, int(count / 15))
            phase = {"linear_prior": .2, "linear_residual": .5, "direct200": .8, "d9": .65}[method]
            result.update(markevery=(min(count - 1, int(step * phase)), step),
                          markersize=2.6 if method == "linear_prior" else 2.8)
    return result


def _times_font_path():
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font

    def is_regular_times(path):
        return (font_manager.FontProperties(fname=str(path)).get_name() == "Times New Roman"
                and FT2Font(str(path)).style_flags == 0)

    override = os.environ.get("PMSM_TIMES_FONT")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError("PMSM_TIMES_FONT must name a locally licensed Times New Roman regular font file")
        if not is_regular_times(path):
            raise ValueError("PMSM_TIMES_FONT must contain regular Times New Roman, not a bold, italic or substitute font")
        return path
    candidates = [Path("C:/Windows/Fonts/times.ttf")]
    try:
        candidates.append(Path(font_manager.findfont(
            font_manager.FontProperties(family="Times New Roman", style="normal", weight="normal"),
            fallback_to_default=False)))
    except ValueError:
        pass
    for path in candidates:
        if path.is_file() and is_regular_times(path):
            return path
    return None


def panel_font():
    from matplotlib import font_manager

    path = _times_font_path()
    exact = path is not None
    if path is None:
        path = Path(font_manager.findfont(font_manager.FontProperties(family="DejaVu Serif"),
                                          fallback_to_default=False))
    properties = font_manager.FontProperties(fname=str(path))
    return properties, dict(font_family=properties.get_name(), font_file=str(path.resolve()),
        font_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), font_redistributed=False,
        times_new_roman=exact, font_fallback_used=not exact,
        font_status="Times New Roman installed locally" if exact else
        "DejaVu Serif substitute; not exact Times New Roman template typography")


def add_panel_labels(fig, axes, insertion_width_mm):
    """Center letters below the visible ticks/x label, at final inserted 8 pt."""
    properties, receipt = panel_font()
    scale = insertion_width_mm / (fig.get_figwidth() * 25.4)
    native_size = 8. / scale
    properties.set_size(native_size)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = []
    for index, ax in enumerate(axes):
        box = ax.get_position()
        tick_length = max([0., *[tick.tick1line.get_markersize() for tick in ax.xaxis.get_major_ticks()
                                if tick.tick1line.get_visible()]])
        bottoms = [box.y0 - tick_length / (72 * fig.get_figheight())]
        for artist in [ax.xaxis.label, ax.xaxis.offsetText, *ax.get_xticklabels()]:
            if artist.get_visible() and artist.get_text():
                bottoms.append(artist.get_window_extent(renderer).transformed(fig.transFigure.inverted()).y0)
        top = min(bottoms) - 2. / (72 * fig.get_figheight())
        text = fig.text((box.x0 + box.x1) / 2, top, f"({chr(97 + index)})",
                        ha="center", va="top", fontproperties=properties, color="black")
        labels.append(dict(text=text.get_text(), axis_index=fig.axes.index(ax),
                           position=list(text.get_position()), native_font_points=native_size,
                           final_font_points=8., insertion_scale=scale,
                           final_include_width_mm=insertion_width_mm))
    return {**receipt, "panel_labels": labels}


def write_style_receipt(directory, stem, panel_receipt=None):
    """Record fresh styling separately from scientific/result provenance."""
    path = Path(directory) / "plot-style-receipt.json"
    value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "schema": "pmsm-fresh-plot-style-v011", "revision": "v011",
        "scope": "Fresh supplied results with v011 visual conventions; not an archived-figure reproduction",
        "exact_archive_layout": False, "font_redistributed": False, "figures": {},
    }
    value["figures"][stem] = {"paper_figure": PAPER_FIGURE_NUMBERS[stem],
        **(panel_receipt or {"panel_labels": []})}
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
