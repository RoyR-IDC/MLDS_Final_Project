# Tile-Permutation Dogs vs Cats

This repository studies Dogs vs Cats classification when each image is split into
square tile grids and the tiles are permuted before classification.

The official workflow is notebook-first. Core logic lives in importable `src`
modules, while the notebooks orchestrate experiments and display saved tables
and figures.

## Setup

```bash
pip install -r requirements.txt
```

The labeled Kaggle training images are expected at:

```text
data/dogs-vs-cats/train
```

Filenames should follow the Kaggle format, for example `cat.123.jpg` and
`dog.456.jpg`. The unlabeled `test1` folder is not used for validation accuracy.

## Official Notebook Workflow

Run the notebooks in this order:

1. `src/notebooks/part1_solution.ipynb`
2. `src/notebooks/part2_solution.ipynb`
3. `src/notebooks/part3_solution.ipynb`

Each notebook has the same structure:

- introduction and notebook topic explanation
- global/external imports
- local imports
- global definitions
- data loading
- experiment sections

Experiment orchestration lives in the notebooks. Reusable training,
preprocessing, plotting, result, and metric helpers stay in `src` modules.
The project does not use YAML configs for the official workflow.

## Outputs

CSV results are saved under `outputs/results/`:

- `part1_raw_results.csv`
- `part1_aggregated_results.csv`
- `part1_permutations.csv`
- `part2_raw_results.csv`
- `part2_aggregated_results.csv`
- `permutation_metrics.csv`
- `metric_accuracy_joined.csv`
- `metric_accuracy_correlations.csv`

Figures are saved under `outputs/figures/`:

- `part1_accuracy_vs_tiles.png`
- `part2_ablation_comparison.png`
- `part3_*_vs_accuracy.png`

## Tests

```bash
pytest
```

Some tests and all full training runs require `torch` and `torchvision`.
