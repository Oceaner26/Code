# Ocean Diviner — Source Code and Experiment Guide

**This repository provides a cleaned, English-documented release of the Ocean Diviner research code (demo version).** It retains the original REMUS simulation, diffusion-based action proposals, TD3 candidate evaluation, and S-Surface control pipeline. Standalone scripts outside the main execution paths have been organized, while the retained source comments and docstrings have been translated or streamlined. **Executable Python syntax trees (excluding documentation strings) are unchanged.** The cleanup preserves the control-law equations, neural-network architecture, reward rules, acoustic formulas, checkpoint tensors, and default parameters.

[Original source settings](docs/ORIGINAL_SETTINGS.md) · [Cleanup log](docs/CLEANUP_LOG.md) · [File policy](docs/FILE_POLICY.md) · [Known limitations](docs/KNOWN_LIMITATIONS.md) · [Validation](docs/VALIDATION.md)

## Core control pipeline

The entry point remains `data_collection/train2.py`:

```text
Current REMUS observation (15 dimensions for two AUVs)
    → original conditional action-diffusion sampler
    → K = 5 candidate action sequences (default)
    → first-action scoring by the TD3 twin critics
    → select action at the requested Q-value rank (default rank 3)
    → high-level depth / heading / propeller references
    → original S-Surface control and 6-DoF REMUS simulation
    → shared IoUT environment, reward and next observation
```

The original evaluation default is **`--eval_policy_rank 3`**, not rank 1. Set `--eval_policy_rank 1` to select the best-valued proposal or `--eval_policy_rank 0` for the source's normal training branch. Each high-level decision executes **only the first action** of the selected diffusion sequence.

## Files retained

```text
Ocean_Diviner_Clean_English/
├── data_collection/
│   ├── train2.py                 # original high-level evaluation/training entry
│   ├── train_diffusion.py       # original alternate training entry
│   ├── eval_td3.py              # original TD3 evaluation entry
│   ├── diffusion_model.py       # original diffusion network and trainer
│   ├── td3.py                   # original actor, twin critics and replay buffer
│   ├── env.py                   # original observation, reward, service and acoustic model
│   ├── usv_usbl.py              # environment's actual USBL dependency
│   ├── record.py, terrain.py, jonswap_wave_current.py
│   ├── terrain.npy              # unchanged original terrain asset
│   ├── diffusion_model.pth      # unchanged original checkpoint
│   └── models_ddpg/            # episode-200 checkpoints used by original loader
├── PVS/                         # complete third-party simulator package and attribution
├── tools/
│   ├── run_original.py          # optional CPU checkpoint mapping; unchanged launcher
│   └── verify_preservation.py  # post-cleanup file and algorithm-integrity checks
├── docs/                         # usage, limitations, cleanup and validation
└── provenance/                   # SHA-256 and executable-AST fingerprints
```

The cleanup removes **12 standalone Python files**, primarily earlier plotting utilities, serialized-run extraction scripts, dataset-merging tools, and an alternative USBL implementation. None is imported by the retained main entry points. Their paths and the cleanup rationale are documented in [CLEANUP_LOG](docs/CLEANUP_LOG.md). The complete PVS third-party package is retained to maintain import compatibility and preserve upstream attribution.

## Installation

Install a PyTorch build appropriate to your CPU/GPU environment, followed by the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Python 3.10+ is suggested. `tools/run_original.py` adds `PVS/src` to the import path without rewriting the vehicle simulator. Run commands from the repository root.

## Running the original pipeline

On a CUDA-enabled system, the original default ranked evaluation is:

```bash
python tools/run_original.py train2
```

For a short, CPU-only **software check** of the same source logic:

```bash
python tools/run_original.py --cpu-map-location train2 --episode_length 3
```

The CPU option maps tensors saved on CUDA to CPU at load time without changing checkpoint values or controller formulas. The three-second example provides a quick check of the source entry point and loading workflow; choose the full task duration and the appropriate experimental settings for quantitative evaluation.

To run the source's normal training branch rather than rank-based evaluation:

```bash
python tools/run_original.py --cpu-map-location train2 \
    --eval_policy_rank 0 --episode_length 3
```

To select the best-Q diffusion proposal at evaluation time:

```bash
python tools/run_original.py --cpu-map-location train2 \
    --eval_policy_rank 1 --episode_length 3
```

The separate `train_diffusion.py` entry uses a prepared training dataset. Set up the input data in the format and location expected by that script before running diffusion-model training. Dataset organization and the retained file structure are described in the accompanying documentation.

## Verify preservation of the scientific implementation

```bash
python tools/verify_preservation.py
```

The verifier checks the SHA-256 hashes of retained source files and assets against the packaged manifest, compares each retained original Python file's executable AST with its pre-cleanup fingerprint (excluding docstrings), and checks for remaining Chinese `#` comments. This provides an integrity check for the published source package; experiment-level evaluation uses the relevant configurations, checkpoints, input data, and recorded metrics.



## Research usage and publication notes

This release brings together the preserved scientific implementation, its runnable entry points, configuration guidance, and source-integrity checks in a consistent English-language repository. When reporting an experiment, record the exact command, runtime configuration, random seed, input dataset, checkpoint version, and output logs so that the experimental setting and resulting metrics can be examined together.

Before publication or redistribution, confirm the applicable rights for the research code, checkpoints, and terrain data, and retain the original PVS licence and author notices. See the [Publication checklist](docs/PUBLICATION_CHECKLIST.md).
