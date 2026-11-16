# Dataset attribution and license scope

Checked on November 16, 2026. Raw datasets are acquired externally. Source licenses remain applicable to those materials and are not replaced by a license for independently written code.

## IPMSM

Source: [Zirui24/lptn_informed_LSTM](https://github.com/Zirui24/lptn_informed_LSTM/tree/98e4566b5fb7c70499996fda18dd73179ec16509), commit 98e4566b5fb7c70499996fda18dd73179ec16509, June 5, 2024. The selected inputs are dataset/id_0.csv through dataset/id_15.csv; dataset/temperature.csv is not used.

Credit: Zirui Liu, 2024. Related work: [Hybrid Thermal Modeling with LPTN-Informed Neural Network for Multi-Node Temperature Estimation in PMSM](https://doi.org/10.1109/TPEL.2024.3409388).

The pinned [root LICENSE](https://github.com/Zirui24/lptn_informed_LSTM/blob/98e4566b5fb7c70499996fda18dd73179ec16509/LICENSE) declares MIT, copyright 2024 Zirui Liu. The README declares MIT and describes the measurements, but the license text addresses software and documentation; a distinct dataset/database grant is not explicit. No separate license was found in the dataset directory. This repository therefore links to the original data rather than asserting that raw-data redistribution has been separately established. This is an identified scope ambiguity, not a claim that redistribution is forbidden. If upstream software is reused, preserve its MIT copyright and permission notice.

## Paderborn

Source: [Electric Motor Temperature, version 3](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature/versions/3), owner wkirgsn, [version DOI 10.34740/KAGGLE/DSV/2161054](https://doi.org/10.34740/KAGGLE/DSV/2161054), April 26, 2021. Input: measures_v2.csv.

Credit: Wilhelm Kirchgässner, Oliver Wallscheid, and Joachim Böcker; LEA, Paderborn University. The source asks users to cite [Estimating Electric Motor Temperatures with Deep Residual Machine Learning](https://ieeexplore.ieee.org/abstract/document/9296842).

Kaggle declares [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Redistribution requires appropriate attribution, license information, indication of changes, and applicable ShareAlike conditions for adapted licensed material. This does not automatically relicense independently written code merely because it reads the data. The indexed official page and preserved official API metadata confirm the declaration. A fresh API view request failed; the DataCite record confirms identity but has no rightsList or ordinal-version field. Exact version and hashes are fixed by the original acquisition records.

## RWTH

Source: [Drive Cycle Measurement Data with rotor and stator temperature from a 48V dual three-phase permanent magnet radial flux synchronous machine](https://zenodo.org/records/15599966), [version DOI 10.5281/zenodo.15599966](https://doi.org/10.5281/zenodo.15599966), June 5, 2025. Input: DriveCycles.zip, with final/001.csv through final/038.csv. Use the version record rather than concept DOI15599965.

Credit: Anna Isabel Ramones, Samuel Kieling, Michael Engels, and Jakob Andert. License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Preserve attribution and license information and indicate modifications if sharing. The registered [DataCite metadata](https://api.datacite.org/dois/10.5281/zenodo.15599966) freshly confirms this license; a direct Zenodo API request returned403. Archive hashes come from the preserved acquisition receipt and a local byte check.

## Integrity scope

The [data registry](../data/registry.json) supplies stable source links, file sizes, SHA-256 and, for IPMSM, Git-blob SHA-1. Local bytes were checked against those identities; no new dataset download was performed for this release. Permissions beyond the source-declared license scope are not independently guaranteed. Do not publish raw data under a blanket code license or include credentials in acquisition instructions.
