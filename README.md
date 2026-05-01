# Kaggle Dogs vs Cats downloader

This repo includes a small helper script to download and extract the Kaggle "Dogs vs Cats" competition data into a local `data/dogs-vs-cats` folder.

Usage

1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Provide Kaggle credentials either by setting the environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`, or by placing `kaggle.json` in `~/.kaggle/`.

3. Run the downloader script from the repository root:

```bash
python3 scripts/download_kaggle_dogs_vs_cats.py
```

The script will download the competition files and extract them into `data/dogs-vs-cats`.

Experiments

1. Install dependencies (or create the conda env):

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
conda activate mlds_dogs_vs_cats
```

2. Run a baseline experiment (example):

```bash
python scripts/train_experiment.py configs/experiments.yaml
```

3. Run the improved experiment (mixup + stronger augmentation):

```bash
python scripts/train_experiment.py configs/experiments_improved.yaml
```

4. Generate plots from results:

```bash
python -m src.utils.plots --summary results/tiles_experiment/summary.csv --out results/tiles_experiment/plots --n_permutations 5
```

Results and artifacts (CSV, checkpoints, plots) are written to the `results/` folder configured in the YAML files.
