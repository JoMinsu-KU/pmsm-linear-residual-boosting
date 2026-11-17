"""Redraw Figures 4 and 6 with print-safe styles and official subfigure labels.

The original asset name for paper Figure 4 is figure5-temperature-trajectories.
Use scripts/render_reference_v011.py to render the saved inputs.
"""
from pathlib import Path
from collections import Counter
from io import BytesIO
import argparse
import csv
import hashlib
import json
import os
import shutil

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib._tight_bbox
from matplotlib.transforms import Bbox
from matplotlib.ticker import FixedLocator
from matplotlib.font_manager import FontProperties, findfont
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PACKAGED = (HERE / 'data').is_dir()
BASE = HERE if PACKAGED else HERE.parent
STYLES = {
    'target': dict(color='#222222', linestyle='-', marker=''),
    'linear_prior': dict(color='#777777', linestyle=':', marker='D', markerfacecolor='white'),
    'linear_residual': dict(color='#0072B2', linestyle=(0, (3, 1.1, .8, 1.1)), marker='o', markerfacecolor='#0072B2'),
    'direct200': dict(color='#D55E00', linestyle='--', marker='s', markerfacecolor='white'),
    'd9': dict(color='#009E73', linestyle=':', marker='^', markerfacecolor='white'),
}
METHODS = ('linear_residual', 'direct200', 'd9')
LABELS = {'linear_residual': 'Linear residual boosting', 'direct200': 'Matched direct boosting', 'd9': 'Selected direct boosting'}
COHORTS = (('ipmsm', 'calibration'), ('ipmsm', 'pilot'), ('ipmsm', 'confirmation'), ('paderborn', 'development'), ('paderborn', 'confirmation'))
ROLES = {'calibration': 'Calibration', 'pilot': 'Pilot evaluation', 'development': 'Development', 'confirmation': 'Historical evaluation'}
NAMES = ('figure5-temperature-trajectories', 'figure6-initialization-segments')
FINAL_WIDTH_MM = 177.53


def times_font_path():
    """Locate a licensed local Times New Roman installation; never bundle it."""
    override = os.environ.get('TIMES_NEW_ROMAN_FONT')
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError('TIMES_NEW_ROMAN_FONT must name a local Times New Roman regular TTF file')
        return path
    windows = Path('C:/Windows/Fonts/times.ttf')
    if windows.is_file():
        return windows
    return Path(findfont(FontProperties(family='Times New Roman', style='normal', weight='normal'), fallback_to_default=False))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def trajectories(rows, examples):
    """Original v009 layout; saved values are plotted in their original order."""
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9,
                         'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0), sharex='col')
    styles = {'target': ('Measured rotor', '#222222', '-', 1.15),
              'linear_prior': ('Ridge prior', '#777777', '--', 1.),
              'linear_residual': ('Linear residual boosting', '#0072B2', '-', 1.0),
              'direct200': ('Matched direct boosting', '#D55E00', ':', 1.05)}
    data_lines = []
    for column, record in enumerate(examples):
        selected = [r for r in rows if r['dataset'] == record['dataset'] and int(r['profile']) == record['profile']]
        assert len(selected) == record['n']
        index = np.array([int(r['within_profile_row']) for r in selected])
        y = np.array([float(r['target_c']) for r in selected])
        for key, (label, color, style, width) in styles.items():
            v = y if key == 'target' else np.array([float(r[key + '_c']) for r in selected])
            line, = axes[0, column].plot(index, v, label=label, color=color, ls=style, lw=width)
            data_lines.append((key, line))
            if key != 'target':
                line, = axes[1, column].plot(index, v-y, color=color, ls=style, lw=width)
                data_lines.append((key, line))
        axes[0, column].set_title(('Paderborn' if record['dataset'] == 'paderborn' else 'IPMSM') + ', profile ' + str(record['profile']) + '\n' + ('Median-near improvement' if column == 0 else 'Largest historical degradation'))
        axes[1, column].axhline(0, color='black', lw=.6)
        axes[1, column].set_xlabel('Within-profile source-row index')
        for ax in axes[:, column]:
            ax.grid(alpha=.18)
            ax.set_xlim(int(index[0]), int(index[-1]))
        axes[1, column].text(.03, .04, 'RMSE: ridge %.2f, residual %.2f, direct %.2f °C' % (record['linear_prior_rmse_c'], record['linear_residual_rmse_c'], record['direct200_rmse_c']),
                            transform=axes[1, column].transAxes, fontsize=7.3, bbox=dict(facecolor='white', alpha=.9, edgecolor='none'))
    axes[0, 0].set_ylabel('Rotor temperature (°C)')
    axes[1, 0].set_ylabel('Prediction minus target (°C)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.0), ncol=2, frameon=False)
    fig.subplots_adjust(top=.82, bottom=.105, left=.09, right=.99, hspace=.2, wspace=.22)
    return fig, data_lines


def segments(rows):
    """Original v009 panel geometry and shared dataset scales."""
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.2), sharex=True)
    colors = {'linear_residual': '#0072B2', 'direct200': '#D55E00', 'd9': '#009E73'}
    markers = {'linear_residual': 'o', 'direct200': 's', 'd9': '^'}
    data_lines = []
    for idx, (ds, role) in enumerate(COHORTS):
        ax = axes.flat[idx]
        selected = [r for r in rows if r['dataset'] == ds and r['role'] == role]
        for method in METHODS:
            current = sorted((r for r in selected if r['method'] == method), key=lambda r: int(r['fifth']))
            line, = ax.plot([int(r['fifth']) for r in current], [float(r['macro_difference']) for r in current],
                            marker=markers[method], color=colors[method], lw=1.4, ms=4.5, label=LABELS[method])
            data_lines.append((method, line))
        count = int(selected[0]['profiles'])
        ax.set_title(('IPMSM' if ds == 'ipmsm' else 'Paderborn') + '\n' + ROLES[role] + f' (n={count})', fontsize=8.5)
        ax.axhline(0, color='black', lw=.8)
        ax.grid(axis='y', alpha=.18)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlim(.75, 5.25)
        ax.tick_params(axis='x', labelbottom=True)
        ax.set_xlabel('Retained-row fifth')
        if idx in (0, 3):
            ax.set_ylabel('Macro-RMSE change (°C)')
    for ds, indices in [('ipmsm', [0, 1, 2]), ('paderborn', [3, 4])]:
        values = [float(r['macro_difference']) for r in rows if r['dataset'] == ds]
        lower, upper = min(0., min(values)), max(0., max(values))
        padding = max(.03, .12 * (upper-lower))
        for i in indices:
            axes.flat[i].set_ylim(lower-padding, upper+padding)
    axes.flat[5].axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[5].legend(handles, labels, loc='upper left', frameon=False, fontsize=8.5,
                        handlelength=1.5, handletextpad=.5, borderaxespad=0., bbox_to_anchor=(-.02, .91))
    axes.flat[5].text(0., .53, 'Change: channel-wise\nminus ambient.\nNegative: lower RMSE.\n\nShared y scale per dataset.\nFifths count rows, not time.', transform=axes.flat[5].transAxes, va='top', fontsize=8.5, linespacing=1.4)
    fig.subplots_adjust(left=.11, right=.985, bottom=.10, top=.89, wspace=.37, hspace=.69)
    return fig, data_lines


def axis_state(fig):
    return [dict(position=list(ax.get_position().bounds), xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()),
                 xticks=ax.get_xticks().tolist(), yticks=ax.get_yticks().tolist(), title=ax.get_title(),
                 xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel()) for ax in fig.axes]


def text_labels(fig):
    return Counter(artist.get_text() for artist in fig.findobj(matplotlib.text.Text) if artist.get_text())


def baseline_pdf(fig, tight):
    """Capture the original PDF renderer's actual crop, including its padding."""
    captured = []
    original = matplotlib._tight_bbox.adjust_bbox
    def capture(figure, bbox_inches, fixed_dpi=None):
        captured.append(tuple(bbox_inches.bounds))
        return original(figure, bbox_inches, fixed_dpi)
    matplotlib._tight_bbox.adjust_bbox = capture
    stream = BytesIO()
    try:
        fig.savefig(stream, format='pdf', bbox_inches='tight' if tight else None)
    finally:
        matplotlib._tight_bbox.adjust_bbox = original
    return stream.getvalue(), Bbox.from_bounds(*captured[-1]) if tight else None


def modify(fig, data_lines, is_trajectory):
    phases = {'linear_prior': .2, 'linear_residual': .5, 'direct200': .8}
    for method, line in data_lines:
        line.set(**STYLES[method])
        if method == 'target':
            continue
        line.set_markeredgecolor(STYLES[method]['color'])
        line.set_markeredgewidth(.75)
        if is_trajectory:
            step = max(1, int(len(line.get_xdata())/15))
            line.set_markevery((max(1, int(step*phases[method])), step))
            line.set_markersize(2.8 if method != 'linear_prior' else 2.6)
    if is_trajectory:
        fig.legends[0].remove()
        handles, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.0), ncol=2, frameon=False, handlelength=3.4)
    else:
        ax = fig.axes[5]
        ax.get_legend().remove()
        handles, labels = fig.axes[0].get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper left', frameon=False, fontsize=8.5, handlelength=2.3,
                  handletextpad=.5, borderaxespad=0., bbox_to_anchor=(-.02, .91))


def pdf_info(source):
    with fitz.open(stream=source, filetype='pdf') if isinstance(source, bytes) else fitz.open(source) as doc:
        page = doc[0]
        return dict(mediabox=list(page.mediabox), cropbox=list(page.cropbox), text=page.get_text(),
                    image_count=len(page.get_images()), font_types=[f[2] for f in page.get_fonts()])


def panel_labels(fig, is_trajectory, bbox):
    """Add centered labels below each panel without moving or resizing axes.

    The main manuscript inserts both figures at its 177.53 mm text width. Native
    font sizes compensate for that scale, making the final labels exactly 8 pt.
    Only lower-row tick/axis-label padding changes to free footer space.
    """
    font_path = times_font_path()
    font = fitz.Font(fontfile=str(font_path))
    native_width_pt = bbox.width * 72 if bbox is not None else fig.get_figwidth() * 72
    scale = (FINAL_WIDTH_MM / 25.4 * 72) / native_width_pt
    size = 8 / scale
    font_properties = FontProperties(fname=str(font_path), size=size)
    if is_trajectory:
        lower = fig.axes[2:4]
        for ax in lower:
            ax.tick_params(axis='x', pad=1.)
            ax.xaxis.labelpad = 2.
        padding = dict(lower_tick_pad_before_pt=3.5, lower_tick_pad_after_pt=1.,
                       lower_xlabel_pad_before_pt=4., lower_xlabel_pad_after_pt=2.)
        active = fig.axes[:4]
    else:
        lower = fig.axes[3:5]
        for ax in lower:
            ax.tick_params(axis='x', pad=2.5)
            ax.xaxis.labelpad = 3.
        padding = dict(lower_tick_pad_before_pt=3.5, lower_tick_pad_after_pt=2.5,
                       lower_xlabel_pad_before_pt=4., lower_xlabel_pad_after_pt=3.)
        active = fig.axes[:5]
    # Measure actual PDF text after the small padding adjustments. The font
    # metrics used below are read from the same TTF embedded by Matplotlib.
    interim = BytesIO()
    fig.savefig(interim, format='pdf', bbox_inches=bbox, dpi=240 if is_trajectory else 250)
    with fitz.open(stream=interim.getvalue(), filetype='pdf') as document:
        spans = [span for block in document[0].get_text('dict')['blocks']
                 for line in block.get('lines', []) for span in line['spans']]
    crop_top_inches = bbox.y1 if bbox is not None else fig.get_figheight()
    crop_left_inches = bbox.x0 if bbox is not None else 0.
    labels = []
    for idx, ax in enumerate(active):
        position = ax.get_position()
        center_x = (position.x0 + position.x1) / 2
        center_pt = (center_x * fig.get_figwidth() - crop_left_inches) * 72
        axis_bottom_pt = (crop_top_inches - position.y0 * fig.get_figheight()) * 72
        xlabel = ax.get_xlabel()
        if xlabel:
            matching = [span for span in spans if span['text'] == xlabel and span['bbox'][1] > axis_bottom_pt]
            closest = min(matching, key=lambda span: abs((span['bbox'][0]+span['bbox'][2])/2-center_pt) + abs(span['bbox'][1]-axis_bottom_pt))
            below_y_pt = closest['bbox'][3] + 1.
            xlabel_bbox = list(closest['bbox'])
        else:
            below_y_pt = axis_bottom_pt + 5.5
            xlabel_bbox = None
        baseline_pt = below_y_pt + font.ascender * size
        figure_y = (crop_top_inches - baseline_pt / 72) / fig.get_figheight()
        label = f'({chr(ord("a")+idx)})'
        fig.text(center_x, figure_y, label, ha='center', va='baseline', fontproperties=font_properties)
        labels.append(dict(label=label, axis_index=idx, axis_center_x_pt=center_pt,
                           axis_bottom_y_pt=axis_bottom_pt, xlabel_bbox_pt=xlabel_bbox,
                           planned_top_y_pt=below_y_pt, planned_baseline_y_pt=baseline_pt))
    return dict(font_file=str(font_path), font_file_sha256=sha(font_path), font_redistributed=False,
                font_family='Times New Roman', final_font_size_pt=8., native_font_size_pt=size,
                insertion_width_mm=FINAL_WIDTH_MM, insertion_scale=scale, padding_adjustments=padding, labels=labels)


def export(fig, data_lines, name, output, previous):
    is_trajectory = name == NAMES[0]
    original_arrays = [(x, line.get_xdata().copy(), line.get_ydata().copy()) for x, line in data_lines]
    old_pdf, bbox = baseline_pdf(fig, is_trajectory)
    old_axes, old_labels = axis_state(fig), text_labels(fig)
    # An explicit crop skips the preliminary tight-layout renderer pass. Freeze
    # the original locator outputs so backend crop transforms cannot retick axes.
    for ax, state in zip(fig.axes, old_axes):
        ax.xaxis.set_major_locator(FixedLocator(state['xticks']))
        ax.yaxis.set_major_locator(FixedLocator(state['yticks']))
    old_info = pdf_info(old_pdf)
    original_pdf = previous / 'figures' / f'{name}.pdf' if previous else None
    if original_pdf and original_pdf.exists():
        archived = pdf_info(original_pdf)
        assert old_info['mediabox'] == archived['mediabox'], (name, 'baseline page box differs', old_info, archived)
        assert old_info['text'] == archived['text'], (name, 'baseline text differs')
    modify(fig, data_lines, is_trajectory)
    assert old_axes == axis_state(fig)
    assert old_labels == text_labels(fig)
    labels_receipt = panel_labels(fig, is_trajectory, bbox)
    assert old_axes == axis_state(fig)
    for (method, x, y), (current, line) in zip(original_arrays, data_lines):
        assert method == current
        np.testing.assert_array_equal(x, line.get_xdata())
        np.testing.assert_array_equal(y, line.get_ydata())
    out = output / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(out / f'{name}.{extension}', bbox_inches=bbox, dpi=240 if is_trajectory else 250)
    new_info = pdf_info(out / f'{name}.pdf')
    assert old_info['mediabox'] == new_info['mediabox']
    assert old_info['cropbox'] == new_info['cropbox']
    added_labels = [entry['label'] for entry in labels_receipt['labels']]
    assert Counter(old_info['text'].splitlines()) + Counter(added_labels) == Counter(new_info['text'].splitlines())
    assert new_info['image_count'] == 0, 'PDF must remain vector artwork'
    with Image.open(out / f'{name}.png') as im:
        rgb = np.asarray(im.convert('RGB'))
        channels_equal = np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(rgb[:, :, 1], rgb[:, :, 2])
        png_size = list(im.size)
    # Inspect rendered PDF text; locator artists outside limits are not drawn.
    overflow = []
    with fitz.open(out / f'{name}.pdf') as document:
        page = document[0]
        embedded_fonts = [f[3] for f in page.get_fonts()]
        assert any('TimesNewRoman' in font for font in embedded_fonts)
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    x0, y0, x1, y1 = span['bbox']
                    if x0 < -.2 or x1 > page.rect.width+.2 or y0 < -.2 or y1 > page.rect.height+.2:
                        overflow.append(span['text'])
                    if span['text'] in added_labels:
                        entry = next(item for item in labels_receipt['labels'] if item['label'] == span['text'])
                        assert 'TimesNewRoman' in span['font']
                        assert abs(span['size'] * labels_receipt['insertion_scale'] - 8.) < 1e-5
                        assert abs((x0+x1)/2 - entry['axis_center_x_pt']) < .1
                        assert y0 > entry['axis_bottom_y_pt']
                        if entry['xlabel_bbox_pt'] is not None:
                            assert y0 > entry['xlabel_bbox_pt'][3]
                        entry.update(rendered_bbox_pt=list(span['bbox']), rendered_font=span['font'],
                                     rendered_native_size_pt=span['size'], rendered_final_size_pt=span['size'] * labels_receipt['insertion_scale'])
        assert all('rendered_bbox_pt' in entry for entry in labels_receipt['labels'])
    assert not overflow, (name, 'text overflow', overflow)
    receipt = dict(status='PASS', data_lines=len(data_lines), line_vertices=sum(len(x) for _, x, _ in original_arrays),
                   every_plotted_value_unchanged=True, axis_positions_limits_ticks_unchanged=True, labels_and_plot_order_unchanged=True,
                   original_pdf_text_preserved=True, added_subfigure_labels=labels_receipt, page_boxes_unchanged=True, original_figure_inches=fig.get_size_inches().tolist(),
                   mediabox=new_info['mediabox'], cropbox=new_info['cropbox'], frozen_crop_inches=list(bbox.bounds) if bbox is not None else None,
                   original_axes=old_axes, new_axes=axis_state(fig), raster_images_in_pdf=0, png_all_rgb_channels_equal=bool(channels_equal),
                   original_color_palette_preserved=True, print_distinction_by_dashes_and_markers=True,
                   png_size=png_size, text_overflow=overflow, baseline_matches_archived_pdf=bool(original_pdf and original_pdf.exists()),
                   original_pdf_sha256=sha(original_pdf) if original_pdf and original_pdf.exists() else None,
                   assets={f'{name}.{ext}': sha(out / f'{name}.{ext}') for ext in ('pdf', 'svg', 'png')})
    plt.close(fig)
    return receipt
