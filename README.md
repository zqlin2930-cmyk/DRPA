# DRPA

**Decoder- and Rank-adaptive Parameter Adaptation** for text-prompted 3D medical image segmentation with VoxTell.

[中文说明](README_zh.md) · [Experiment map](docs/EXPERIMENTS.md) · [Data and weights](docs/ASSETS.md) · [Validation](docs/VALIDATION.md)

This distribution contains the author's available model, training, evaluation, baseline, placement/rank ablation, external-domain and convergence source code. The scientific implementations retain their original layout under `drpa/research/`; a lightweight CLI prepares separate, relocatable run directories. It does not silently replace the published method with a new implementation.

This is a **code release**, not a patient-data or trained-weight release. It includes 128 collected Python source files, source/config provenance, and the VoxTell upstream license. Historical operational scripts and source snapshots are retained for traceability; consult the experiment map before selecting an entrypoint.

## Contents

```text
DRPA/
├── drpa/                       # CLI, workspace preparation, fresh queue
│   ├── research/               # Full collected research source, original layout
│   │   ├── VoxTell/             # Upstream source and Apache-2.0 license
│   │   ├── quality_audit/       # DRPA/PEFT/baseline wrappers and preprocessing
│   │   ├── scripts/             # Baselines, ablations, external evaluation
│   │   └── convergence_continuous_seed20260809/
│   │       ├── code/            # Original continuous-training implementation
│   │       ├── validation_v2/   # Equivalent shared-distance threaded evaluation
│   │       └── plateau_v3/      # Stopping decisions, checks, reporting
│   └── source_inventory.json   # Original/distributed hashes and modifications
├── docs/                       # Reproduction scope and audit records
├── examples/                   # Private asset-map template; synthetic manifest
├── tests/                      # Packaging and CPU scientific checks
├── requirements.txt            # Observed research-runtime package versions
└── pyproject.toml
```

## Installation and inspection

Python 3.12/Linux was used in the source runtime. CUDA training requires PyTorch and a suitable NVIDIA GPU; the full-FP32 FullFT arm was run on a 96-GB RTX PRO 6000. Inspection and workspace preparation use the standard library and can run without a GPU.

```bash
python -m pip install -e .
python -m drpa inspect
python -m drpa --help
```

For research execution, first install the PyTorch 2.8.0 CUDA wheel appropriate to the machine, then:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e drpa/research/VoxTell
```

`docs/environment.observed.json` records the source environment. The pinned requirements are an export of that environment, not a claim that arbitrary Python/CUDA combinations have been tested. Optional upstream visualization/server dependencies are separate in `requirements-optional.txt`.

## Prepare a run directory

Do not execute archived research scripts directly. They intentionally require a prepared workspace so that the original server paths cannot affect a live experiment.

```bash
# A code-only workspace; does not train or access any medical images.
python -m drpa prepare --workspace /absolute/path/to/new_drpa_workspace
python -m drpa doctor --workspace /absolute/path/to/new_drpa_workspace
```

`doctor` reports missing private assets and returns a nonzero exit code until they are supplied. To prepare for the canonical continuous experiment, copy `examples/assets.example.json` to a private `assets.local.json`, replace the example paths with your own authorized data/model locations, and use a **new** destination:

```bash
python -m drpa prepare --workspace /absolute/path/to/drpa_run --assets assets.local.json
python -m drpa doctor --workspace /absolute/path/to/drpa_run
```

Assets are linked into the separate run directory; original data files are not copied into the release. Existing workspaces and source/config files cannot be overwritten by the preparation command. Never upload `assets.local.json` or the prepared run directory: these contain private paths and may acquire patient-level results.

## Continuous convergence experiment

The default fresh queue is DRPA-8 → PD-FT → VoxTell-FullFT. It uses the collected FP32 training implementation, the optimized validation implementation, and the same stopping rule for every model:

- Evaluate at 0, 1500, 3000, 4500, 6000, 7500 and 9000 updates.
- At 9000, stop only if the previous 1500-update interval meets the practical-plateau rule; otherwise extend to 10500.
- At 10500, stop at plateau; extend to 12000 only if at least one metric improves meaningfully with no material regression.
- At 12000, stop at the budget cap and separately record whether plateau was confirmed.

Positive improvement is higher Dice/Surface Dice and lower HD95. Plateau requires all changes to lie strictly inside ±0.10 percentage points for mean Dice, ±0.05 percentage points for Surface Dice, and ±0.05 mm for HD95. Undefined metrics cannot satisfy plateau. Deterioration, conflicting metrics and budget exhaustion are distinct stop reasons.

```bash
# This command actually starts training. Run inside your own durable terminal/session.
python -m drpa convergence --workspace /absolute/path/to/drpa_run
```

The inherited preflight checks 337 train PTIDs/971 visits and 85 validation PTIDs/247 visits, disjoint identities, identical manifests across models, exact prompt embeddings, source hashes, preprocessing, RAM and disk. It does not generate or relax private cohort splits. There is no automatic crash retry or parameter adjustment. Other cohorts require an explicit new experimental protocol.

Each run produces training/validation tables, checkpoints, optimizer/RNG state, stopping decisions and an audit. The final report keeps common 6000/9000-update comparisons and each model's actual stopping endpoint; it does not pad unmeasured checkpoints. The additional `drpa.convergence` supervisor starts every model fresh and does not execute the historical live-process adoption scripts.

## Other experiments

Run a documented source entrypoint in a prepared directory:

```bash
python -m drpa run --workspace /absolute/path/to/drpa_run scripts/train/rank_ablation_50_runner.py -- --help
```

Main baselines, parameter placement, ranks, external frozen evaluation and OASIS few-shot studies have separate historical manifests, checkpoints and provenance gates. Their complete available source is included; the private historical artifacts required by their gates are not fabricated or bundled. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Tests and release checks

```bash
python -m unittest discover -s tests -p 'test_packaging.py' -v
DRPA_RUN_SCIENCE_TESTS=1 python -m unittest discover -s tests -v
python tools/check_release.py
```

The CPU science tests need the research dependencies but do not load pretrained weights or train on patient data. The release audit documents which tests actually ran. A full relocated GPU training run was not performed during packaging because the author's existing training was still active.

## Method and interpretation

Configured trainable segmentation parameters: B1 294,912; DRPA-8 10,969,696; PD-FT/B3-Canonical 81,930,624; VoxTell-FullFT 440,029,541. The text encoder is frozen and excluded from these counts. Archived exploratory B3 variants are not interchangeable with the canonical PD-FT baseline. The historical mainline DRPA AMP run and the later strict-FP32 convergence run are distinct protocols.

Practical plateau is an operational single-seed stopping criterion, not statistical equivalence, a clinical minimum important difference, or proof of permanent convergence. Preprocessing includes the original reference-assisted crop behavior; do not describe this implementation as an independently validated label-free deployment pipeline.

## Attribution and licensing

VoxTell source is included at the recorded upstream commit `8ef0332aec5d5dff925d6f6370386c00334b57ab` with its original license and notices; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). A project-wide open-source license for the author's own additions has not been selected; see [LICENSE.md](LICENSE.md). No manuscript publication status or author list is invented in this release.
