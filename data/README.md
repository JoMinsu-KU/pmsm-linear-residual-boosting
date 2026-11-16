# Data acquisition

Raw measurements are not included. The [registry](registry.json) fixes the original source versions and exact bytes; [attribution and license scope](../docs/DATA_ATTRIBUTION.md) are separate from the repository's original code.

Run from the repository root with Python 3.10 or later. Only the standard library is needed.

```sh
python scripts/acquire_data.py
python scripts/acquire_data.py --dataset paderborn
```

These commands only display URLs, sizes, hashes, and destinations. They create no data files and make no network requests.

An explicit download command permits network access and verified extraction:

```sh
python scripts/acquire_data.py --dataset ipmsm --download
python scripts/acquire_data.py --dataset paderborn --download
python scripts/acquire_data.py --dataset rwth --download
```

Default destinations are data/raw/ipmsm, data/raw/paderborn, and data/raw/rwth. Existing files must match the pinned hash and are never overwritten. Failed downloads or extraction can leave uniquely named .part files for inspection. A retry uses a new partial file; remove obsolete partial files yourself after inspection. ZIP extraction accepts only registered members and checks their sizes and hashes.

Offline verification makes no network requests or filesystem changes:

```sh
python scripts/acquire_data.py --verify-only
python scripts/acquire_data.py --dataset paderborn --verify-only --directory /path/to/paderborn
```

The --directory path is the selected dataset's root, not a parent containing all datasets. It requires a single --dataset. For IPMSM, place id_0.csv through id_15.csv directly there. For Paderborn, place measures_v2.csv or the pinned measures_v2.csv.zip there. For RWTH, place DriveCycles.zip there, or all 38 CSVs under final/.

If a pinned archive exists, its transport hash is checked strictly. If only all extracted members exist, their hashes suffice for offline verification or reuse. No silent version replacement or archive-hash bypass is implemented. Kaggle may recompress a ZIP: a differing ZIP is rejected. In that situation, manually obtain version 3 from the official page, extract measures_v2.csv yourself, and verify that exact CSV in a directory without the mismatching ZIP. CSV contents must match the registry. Do not modify the registry to accept an unexplained mismatch.

Kaggle's pinned file request historically returns a ZIP, not a bare CSV. The audited metadata API returned 404, and Zenodo's metadata API returned 403; payload downloads were not retested when preparing this release. Use the registry's official landing pages for a manual download if necessary. Honor account and access requirements. Never commit credentials, signed redirect URLs, raw data, caches, or .part files.

These commands acquire inputs only. They do not rerun the paper's training or establish standalone experiment reproducibility. Experimental splits are documented separately from acquisition identity.
