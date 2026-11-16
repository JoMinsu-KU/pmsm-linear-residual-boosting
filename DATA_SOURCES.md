# Data sources and acquisition

The repository redistributes no raw measurements. [data/registry.json](data/registry.json) records the exact source versions, download URLs, file sizes, SHA-256 checksums, archive members, and attribution used by the pipeline. Acquire data from the upstream providers under their applicable terms.

| Dataset | Pinned source and attribution | Terms recorded in the registry |
| --- | --- | --- |
| IPMSM | [LPTN-informed LSTM repository](https://github.com/Zirui24/lptn_informed_LSTM/tree/98e4566b5fb7c70499996fda18dd73179ec16509), commit `98e4566b5fb7c70499996fda18dd73179ec16509`; Zirui Liu, 2024; [associated paper](https://doi.org/10.1109/TPEL.2024.3409388) | The repository carries an [MIT license](https://github.com/Zirui24/lptn_informed_LSTM/blob/98e4566b5fb7c70499996fda18dd73179ec16509/LICENSE). A separate dataset-specific grant is not independently established here. |
| Paderborn | [Electric Motor Temperature, version 3](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature/versions/3), [version DOI](https://doi.org/10.34740/KAGGLE/DSV/2161054); Wilhelm Kirchgässner, Oliver Wallscheid, Joachim Böcker, LEA, Paderborn University | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); preserve attribution and applicable ShareAlike terms. |
| RWTH | [48 V drive-cycle measurements, record 15599966](https://zenodo.org/records/15599966), [version DOI](https://doi.org/10.5281/zenodo.15599966); Anna Isabel Ramones, Samuel Kieling, Michael Engels, Jakob Andert, 2025 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); preserve attribution, license notice, and indication of changes. |

## Automatic acquisition

```bash
python scripts/acquire_data.py --dataset all
python scripts/acquire_data.py --dataset all --download
python scripts/acquire_data.py --dataset all --verify-only
```

The first command only prints planned transfers. `--download` explicitly enables HTTPS downloads and extraction; `--verify-only` performs offline verification without extraction. A single dataset can be selected with `--dataset ipmsm`, `paderborn`, or `rwth`. For a custom single-dataset destination:

```bash
python scripts/acquire_data.py --dataset paderborn --directory /path/to/raw/paderborn --download
```

`--directory` is incompatible with `--dataset all`. The reproduction pipeline accepts the common parent directory through `--data-root /path/to/raw`.

Expected transfer sizes are approximately 15.4 MB for IPMSM, 122.2 MB for the Paderborn archive, and 3.8 MB for the RWTH archive. Keeping both archives and extracted files uses approximately 453 MB in total. These decimal sizes exclude generated model and report outputs.

Only registered archive members are accepted. Existing valid files are reused; an existing size or hash mismatch stops processing without overwriting the file. Failed downloads may leave uniquely named `.part.*` files for inspection. No credentials are collected or stored by the acquisition script.

## Manual fallback

Provider access, authentication requirements, or archive compression can change. If the automatic transfer fails, use the pinned upstream landing page and place the original files in this structure:

```text
data/raw/
  ipmsm/
    id_0.csv ... id_15.csv
  paderborn/
    measures_v2.csv
  rwth/
    final/
      001.csv ... 038.csv
```

For IPMSM, obtain the sixteen `dataset/id_<id>.csv` files from the pinned commit. The upstream `temperature.csv` is not used. For Paderborn, select dataset version 3 and extract `measures_v2.csv`. For RWTH, select record **15599966**, download `DriveCycles.zip`, and retain its `final/` directory. Do not substitute a newer version or the RWTH concept DOI for the pinned version.

Alternatively, the exact registered archives may remain at `data/raw/paderborn/measures_v2.csv.zip` and `data/raw/rwth/DriveCycles.zip`; preparation can read them directly. A transport archive can be recompressed without changing its member files. If a newly supplied ZIP fails the registered archive hash, extract it separately and supply only the registered extracted files in a clean dataset directory, then verify them. The verifier accepts all verified extracted members without requiring the transport ZIP. If a mismatched ZIP remains at its registered path, verification still fails even when its extracted members are correct.

Verify before preparation:

```bash
python scripts/acquire_data.py --dataset all --verify-only
```

Hashes refer to exact bytes. Do not resave CSVs through a spreadsheet, change line endings, reorder rows, rename columns, or relax checksums to bypass a mismatch. All model splits and source-row order are fixed by the configuration. The public cohorts are historically exposed and are not a new independent test set.
