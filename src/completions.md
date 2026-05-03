You are working in an existing PyTorch repository for an academic deep learning project on image classification under tile-wise permutations.

Your task is to inspect the repository and fix/complete the implementation so it fully satisfies the authoritative repo prompt. Do not invent a new problem, dataset, or assumptions. The dataset is Kaggle Dogs vs Cats. Images are split into tile grids such as 2x2, 3x3, 4x4, etc., then the tiles are permuted before classification.

Important coding requirements:
- Code must be readable, concise, vectorized where practical, and easy for a student to explain orally.
- Use helper methods where needed.
- Keep functions short and focused.
- Add tests for core utilities where appropriate.
- Use Google-style docstrings for all public classes and functions.
- Follow the existing repository structure and coding style.
- Use the existing src layout.
- Put run YAML configs in the configs folder.
- Use PyTorch and torchvision.
- Use standard model implementations where possible.
- Do not add unnecessary files.
- Do not rely on notebook-only logic for core functionality.
- The notebooks should become thin, reproducible experiment/report interfaces that call src modules.

First, analyze the existing repository structure and produce a concrete implementation plan before editing code.

Then implement the missing pieces.

Required final repository behavior
================================

PART 1 — Baseline experiments
-----------------------------

Implement a complete baseline experiment pipeline that measures how classification accuracy degrades as the number of tiles increases.

Must support at least 3 architecture families:

1. CNN:
   - Example: resnet18, resnet34, or efficientnet_b0 from torchvision.

2. Transformer-based vision model:
   - Example: vit_b_16, swin_t, or another standard torchvision vision transformer model.

3. A meaningfully different third architecture:
   - Example: ConvMixer, MLP-Mixer if already available, or a simple standard/open implementation.
   - Do not invent an unmotivated custom architecture unless necessary.
   - If implementing ConvMixer locally, keep it standard, compact, and well documented.

For each model:
- Train on unpermuted images as the baseline.
- Train on permuted images for multiple tile resolutions, for example:
  - identity / unpermuted
  - 2x2
  - 3x3
  - 4x4
- For each fixed tile resolution, evaluate multiple random permutations and average the results for stability.
- Support multiple seeds/runs per configuration.
- Save per-run and aggregated results to CSV.
- Produce a table of results.
- Produce a plot:
  - accuracy vs number of tiles
  - one curve per model
- Save plots under a reproducible output directory, for example outputs/figures/.
- Save logs/results under outputs/results/ or outputs/logs/.

PART 2 — Performance improvement
--------------------------------

Choose one model from Part 1 and implement a clearly isolated improvement strategy for permuted images.

Acceptable strategies include:
- stronger data augmentation,
- pretraining/fine-tuning,
- auxiliary objective,
- specialized loss,
- architectural tuning within the same model family,
- permutation augmentation during training.

Requirements:
- There must be a direct baseline vs improved comparison.
- The improvement must be controlled by config flags.
- The code must support ablation, for example:
  - baseline
  - augmentation only
  - permutation augmentation only
  - full improved setup
- Results must be directly comparable to Part 1.
- Save CSV results and plots.
- The notebook for Part 2 must call the same training/evaluation utilities instead of duplicating training logic.

PART 3 — Permutation difficulty metric
--------------------------------------

Design and implement a model-agnostic permutation difficulty metric.

Hard constraints:
- The metric must not use classification accuracy.
- The metric must not use labels.
- It may use:
  - the permutation itself,
  - tile coordinates,
  - optional image structure without labels.

Implement at least one primary metric and optionally supporting metrics.

Recommended metric components:
- average spatial displacement,
- normalized displacement,
- adjacency preservation / locality disruption,
- displacement entropy,
- graph-based adjacency disruption.

The metric implementation must:
- work for arbitrary square grids, e.g. 2x2, 3x3, 4x4;
- operate on the exact permutations used in Part 1;
- save metric values to CSV;
- merge metric values with empirical accuracies from Part 1;
- compute correlation statistics:
  - Pearson correlation,
  - Spearman correlation;
- produce plots:
  - metric value vs empirical accuracy,
  - optionally one plot per model or a faceted/grouped plot.

Core modules to add or fix
==========================

Inspect the current src structure first. Then add or modify modules as needed, preferably along these lines:

- src/data/dogs_cats.py
  - dataset discovery
  - train/validation/test split
  - label parsing from filenames
  - transforms
  - dataloader creation

- src/data/tile_permutation.py
  - TilePermutationDataset wrapper
  - split image tensor into tiles
  - apply permutation
  - reconstruct image tensor
  - identity permutation
  - seeded random permutation generation
  - reusable list of permutations per grid/seed

- src/models/factory.py
  - get_model(name, num_classes, pretrained, device, ...)
  - support CNN, transformer, and third architecture
  - replace final classifier heads correctly

- src/training/engine.py
  - train_one_epoch
  - evaluate
  - fit
  - checkpoint saving/loading if useful
  - device handling
  - mixed precision optional but not required

- src/training/metrics.py
  - classification accuracy
  - loss aggregation
  - result row formatting

- src/experiments/part1_baselines.py
  - experiment runner for all models, grids, permutations, seeds
  - save raw CSV and aggregated CSV
  - save accuracy-vs-number-of-tiles plot

- src/experiments/part2_improvement.py
  - controlled baseline vs improved experiment runner
  - ablation configs
  - save CSV and comparison plots

- src/experiments/part3_difficulty.py
  - compute permutation metrics
  - join metrics with Part 1 results
  - compute Pearson/Spearman correlations
  - save CSV and plots

- src/permutation_metrics.py or src/metrics/permutation_difficulty.py
  - average_displacement
  - normalized_average_displacement
  - adjacency_preservation
  - locality_disruption
  - displacement_entropy
  - combined difficulty score if justified

- src/utils/reproducibility.py
  - seed_everything
  - deterministic options

- src/utils/io.py
  - ensure_dir
  - save_json
  - save_csv helpers if useful

- configs/part1_baselines.yaml
- configs/part2_improvement.yaml
- configs/part3_difficulty.yaml

- tests/test_tile_permutation.py
- tests/test_permutation_metrics.py

Notebook requirements
=====================

Update the notebooks so they are not toy demos.

data_and_loaders.ipynb:
- Demonstrate dataset discovery.
- Show class counts.
- Show tile splitting and reconstruction.
- Show identity and random permutation examples.
- Do not contain core logic that belongs in src.

part1_solution.ipynb:
- Load configs/part1_baselines.yaml.
- Run or demonstrate the Part 1 experiment runner.
- Display the aggregated result table.
- Display the accuracy-vs-number-of-tiles plot.
- Clearly show the 3 model families.
- Clearly show repeated permutations and averaged results.

part2_solution.ipynb:
- Load configs/part2_improvement.yaml.
- Run or demonstrate baseline vs improved model.
- Show ablation table.
- Plot baseline vs improved performance.
- Save outputs.

part3_solution.ipynb:
- Load Part 1 results.
- Compute permutation difficulty metrics for the exact permutations used in Part 1.
- Join metrics with empirical accuracy.
- Show correlation table.
- Plot metric value vs accuracy.
- Save outputs.

Implementation details
======================

Tile permutation:
- Input should be image tensors shaped [C, H, W].
- Validate that H and W are divisible by grid_size.
- Use tensor reshaping/permutation rather than slow Python image cropping where possible.
- The permutation should map original tile indices to new tile positions in a documented way.
- Identity permutation must leave the image exactly unchanged.
- Add tests proving:
  - identity permutation returns the same tensor,
  - random permutation preserves tensor shape,
  - invalid grid sizes raise clear errors,
  - metric values are sane for identity vs random permutations.

Training:
- Use a single clean training engine.
- Avoid duplicated loops in notebooks.
- Support CPU and single GPU.
- Use configurable:
  - batch_size,
  - epochs,
  - learning_rate,
  - optimizer,
  - weight_decay,
  - image_size,
  - num_workers,
  - seeds,
  - model names,
  - grid sizes,
  - number of permutations per grid.
- Use small defaults that can run on Colab, but allow larger settings through YAML.
- Save all results with enough metadata:
  - part,
  - model_name,
  - grid_size,
  - num_tiles,
  - permutation_id,
  - seed,
  - train_loss,
  - val_loss,
  - val_accuracy,
  - best_val_accuracy,
  - config name,
  - timestamp or run_id.

Model loading:
- Use torchvision model weights where practical.
- Replace classifier heads correctly.
- Keep num_classes=2.
- Make pretrained configurable.
- Ensure transformer models use the expected input size and normalization.

Part 2 improvement:
- Use a realistic, explainable improvement. A good default is:
  - train with random tile permutations sampled each epoch for the chosen grid sizes,
  - optionally combine with standard image augmentations,
  - fine-tune a pretrained model.
- Implement ablation flags:
  - use_pretrained
  - use_standard_augmentation
  - use_permutation_augmentation
  - freeze_backbone or not, if used.
- The output must make it clear which setting is the baseline and which is improved.

Part 3 metric:
- Implement metrics without using labels or accuracy.
- Then compare the already-computed metric values against empirical accuracy only in the analysis stage.
- Compute Pearson and Spearman correlations using pandas/scipy if available. If scipy is unavailable, implement or gracefully skip Spearman with a clear message.
- Save:
  - permutation_metrics.csv
  - metric_accuracy_joined.csv
  - metric_accuracy_correlations.csv
  - metric_vs_accuracy plots.

Acceptance criteria
===================

The repository is complete only if all of the following are true:

1. Part 1 supports 3 different model families.
2. Part 1 supports identity plus multiple tile grids.
3. Part 1 supports multiple random permutations per grid.
4. Part 1 saves raw and aggregated CSVs.
5. Part 1 creates an accuracy-vs-number-of-tiles plot with one curve per model.
6. Part 2 has a controlled baseline-vs-improved comparison.
7. Part 2 supports ablations via config.
8. Part 2 saves comparable CSVs and plots.
9. Part 3 implements at least one model-agnostic permutation difficulty metric.
10. Part 3 does not use labels or classification accuracy inside the metric.
11. Part 3 computes the metric for the exact permutations used in Part 1.
12. Part 3 joins metrics with empirical accuracy only for analysis.
13. Part 3 saves correlation statistics and plots.
14. All public functions/classes have Google-style docstrings.
15. Core logic is in src modules, not duplicated in notebooks.
16. YAML configs exist under configs/.
17. The code runs on CPU or a single Colab GPU.
18. Tests exist for permutation and metric utilities.
19. The final README or notebook text explains how to reproduce all figures.
20. The final implementation avoids unsupported claims and keeps the methodology explainable.

After implementing:
- Run formatting/linting if available.
- Run the tests.
- Run at least a smoke test with tiny settings to verify the full pipeline executes.
- Report:
  1. files added,
  2. files modified,
  3. commands to reproduce Part 1, Part 2, and Part 3,
  4. where CSVs and plots are saved,
  5. any limitations or assumptions.