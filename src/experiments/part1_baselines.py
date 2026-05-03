"""Part 1 baseline experiment runner."""

from __future__ import annotations

from datetime import datetime
import json
import os
from typing import Dict, List, Optional

import pandas as pd
import torch

from src.data.dogs_cats import build_dataloaders
from src.data.tile_permutation import build_permutation_records
from src.experiments.common import aggregate_accuracy, get_device, load_experiment_samples, plot_accuracy_vs_tiles
from src.models.factory import get_model
from src.training.engine import build_optimizer, fit
from src.utils.config import load_experiment_config, normalize_config
from src.utils.io import ensure_dir, save_csv
from src.utils.reproducibility import seed_everything


class Part1BaselineExperiment:
    """Notebook-friendly runner for Part 1 baseline experiments.

    Args:
        config: Grouped or flat Part 1 experiment configuration.
    """

    def __init__(self, config: Dict) -> None:
        self.config = normalize_config(config)
        self.results_dir = ensure_dir(self.config.get("results_dir", "outputs/results"))
        self.figures_dir = ensure_dir(self.config.get("figures_dir", "outputs/figures"))
        self.run_id = self.config.get("run_id") or datetime.now().strftime("part1_%Y%m%d_%H%M%S")
        self.device = get_device(self.config)

    def load_data(self, seed: Optional[int] = None):
        """Load and split Dogs vs Cats samples for a seed."""

        selected_seed = int(self.config.get("seeds", [0])[0] if seed is None else seed)
        return load_experiment_samples(self.config, seed=selected_seed)

    def load_results(self) -> Dict[str, pd.DataFrame]:
        """Load saved raw and aggregated Part 1 result tables."""

        return {
            "raw": pd.read_csv(os.path.join(self.results_dir, "part1_raw_results.csv")),
            "aggregated": pd.read_csv(os.path.join(self.results_dir, "part1_aggregated_results.csv")),
        }

    def display_outputs(self) -> Dict[str, str]:
        """Return saved output paths for notebook display cells."""

        return {
            "raw_results": os.path.join(self.results_dir, "part1_raw_results.csv"),
            "aggregated_results": os.path.join(self.results_dir, "part1_aggregated_results.csv"),
            "permutations": os.path.join(self.results_dir, "part1_permutations.csv"),
            "accuracy_plot": os.path.join(self.figures_dir, "part1_accuracy_vs_tiles.png"),
        }

    def _permutation_records(self):
        grid_sizes = [int(value) for value in self.config.get("grid_sizes", [1, 2, 3, 4])]
        return build_permutation_records(
            grid_sizes=grid_sizes,
            num_permutations=int(self.config.get("num_permutations", 2)),
            permutation_seed=int(self.config.get("permutation_seed", 42)),
            include_identity=True,
        )

    def run(self) -> pd.DataFrame:
        """Run Part 1 baselines and save CSV/figure artifacts."""

        rows: List[dict] = []
        permutation_records = self._permutation_records()
        metadata_path = os.path.join(self.results_dir, "part1_permutations.csv")
        save_csv([record.__dict__ | {"permutation": json.dumps(record.permutation)} for record in permutation_records], metadata_path)

        for seed in self.config.get("seeds", [0]):
            seed_everything(int(seed), deterministic=bool(self.config.get("deterministic", False)))
            train_samples, val_samples, _ = self.load_data(seed=int(seed))
            for model_name in self.config.get("model_names", ["resnet18", "swin_t", "convmixer"]):
                for record in permutation_records:
                    # The identity grid has only one meaningful permutation.
                    if record.grid_size == 1 and record.permutation_id > 0:
                        continue
                    train_loader, val_loader = build_dataloaders(
                        train_samples,
                        val_samples,
                        image_size=int(self.config.get("image_size", 224)),
                        grid_size=record.grid_size,
                        permutation=record.permutation,
                        seed=int(seed),
                        batch_size=int(self.config.get("batch_size", 32)),
                        num_workers=int(self.config.get("num_workers", 2)),
                        standard_augmentation=False,
                    )
                    model = get_model(
                        model_name,
                        num_classes=int(self.config.get("num_classes", 2)),
                        pretrained=bool(self.config.get("pretrained", False)) and model_name != "convmixer",
                        device=self.device,
                        convmixer_dim=int(self.config.get("convmixer_dim", 256)),
                        convmixer_depth=int(self.config.get("convmixer_depth", 8)),
                    )
                    optimizer = build_optimizer(
                        model,
                        name=self.config.get("optimizer", "adamw"),
                        learning_rate=float(self.config.get("learning_rate", 1e-4)),
                        weight_decay=float(self.config.get("weight_decay", 0.0)),
                    )
                    metrics = fit(
                        model,
                        train_loader,
                        val_loader,
                        epochs=int(self.config.get("epochs", 1)),
                        optimizer=optimizer,
                        criterion=torch.nn.CrossEntropyLoss(),
                        device=self.device,
                        use_amp=bool(self.config.get("use_amp", False)),
                    )
                    rows.append(
                        {
                            "part": "part1",
                            "run_id": self.run_id,
                            "config_name": self.config.get("config_name", "part1_baselines"),
                            "model_name": model_name,
                            "grid_size": record.grid_size,
                            "num_tiles": record.grid_size * record.grid_size,
                            "permutation_id": record.permutation_id,
                            "permutation_seed": record.permutation_seed,
                            "seed": int(seed),
                            **metrics,
                        }
                    )
                    save_csv(rows, os.path.join(self.results_dir, "part1_raw_results.csv"))

        raw_results = pd.DataFrame(rows)
        aggregated = aggregate_accuracy(raw_results, ["model_name", "grid_size", "num_tiles"])
        save_csv(aggregated, os.path.join(self.results_dir, "part1_aggregated_results.csv"))
        plot_accuracy_vs_tiles(aggregated, os.path.join(self.figures_dir, "part1_accuracy_vs_tiles.png"))
        return aggregated


def run_part1(config: Dict) -> pd.DataFrame:
    """Run Part 1 baseline experiments and save CSV/figure artifacts."""

    return Part1BaselineExperiment(config).run()


def main(config_path: str) -> pd.DataFrame:
    """Run Part 1 from a YAML config path."""

    return run_part1(load_experiment_config(config_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    main(args.config)
