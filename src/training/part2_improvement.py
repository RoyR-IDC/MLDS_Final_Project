"""Part 2 controlled improvement and ablation runner."""

from __future__ import annotations

from datetime import datetime
import os
from typing import Dict, List, Optional

import pandas as pd
import torch

from src.evaluation.experiment_results import aggregate_accuracy, get_device, load_experiment_samples
from src.models.factory import get_model
from src.preprocessing.dogs_cats import build_dataloaders
from src.preprocessing.permutations import generate_permutations, identity_permutation
from src.training.engine import build_optimizer, fit
from src.utils.config import load_experiment_config, normalize_config
from src.utils.io import ensure_dir, save_csv
from src.utils.reproducibility import seed_everything


def _plot_ablation_results(aggregated: pd.DataFrame, output_path: str) -> None:
    """Save a baseline-vs-improvement ablation plot.

    Args:
        aggregated: Aggregated Part 2 result table.
        output_path: Destination path for the figure.
    """

    import matplotlib.pyplot as plt

    ensure_dir(os.path.dirname(output_path) or ".")
    fig, ax = plt.subplots(figsize=(8, 5))
    for grid_size, group in aggregated.groupby("grid_size"):
        group = group.sort_values("ablation_name")
        ax.plot(group["ablation_name"], group["mean_best_val_accuracy"], marker="o", label=f"{grid_size}x{grid_size}")
    ax.set_xlabel("Ablation")
    ax.set_ylabel("Best validation accuracy")
    ax.set_title("Baseline vs Improved Ablations")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _ablation_configs(config: Dict) -> List[Dict]:
    """Return configured ablations or the default controlled comparison set.

    Args:
        config: Normalized experiment configuration.

    Returns:
        List of ablation dictionaries.
    """

    default_ablations = [
        {"name": "baseline", "use_pretrained": False, "use_standard_augmentation": False, "use_permutation_augmentation": False},
        {
            "name": "augmentation_only",
            "use_pretrained": False,
            "use_standard_augmentation": True,
            "use_permutation_augmentation": False,
        },
        {
            "name": "permutation_augmentation_only",
            "use_pretrained": False,
            "use_standard_augmentation": False,
            "use_permutation_augmentation": True,
        },
        {"name": "full_improved", "use_pretrained": True, "use_standard_augmentation": True, "use_permutation_augmentation": True},
    ]
    ablation_configs = config.get("ablations", default_ablations)
    return ablation_configs


class Part2ImprovementExperiment:
    """Notebook-friendly runner for Part 2 improvement ablations.

    Args:
        config: Grouped or flat Part 2 experiment configuration.
    """

    def __init__(self, config: Dict) -> None:
        self.config = normalize_config(config)
        self.results_dir = ensure_dir(self.config.get("results_dir", "outputs/results"))
        self.figures_dir = ensure_dir(self.config.get("figures_dir", "outputs/figures"))
        self.run_id = self.config.get("run_id") or datetime.now().strftime("part2_%Y%m%d_%H%M%S")
        self.device = get_device(self.config)

    def load_data(self, seed: Optional[int] = None):
        """Load and split Dogs vs Cats samples for a seed."""

        selected_seed = int(self.config.get("seeds", [0])[0] if seed is None else seed)
        samples = load_experiment_samples(self.config, seed=selected_seed)
        return samples

    def load_results(self) -> Dict[str, pd.DataFrame]:
        """Load saved raw and aggregated Part 2 result tables."""

        result_tables = {
            "raw": pd.read_csv(os.path.join(self.results_dir, "part2_raw_results.csv")),
            "aggregated": pd.read_csv(os.path.join(self.results_dir, "part2_aggregated_results.csv")),
        }
        return result_tables

    def display_outputs(self) -> Dict[str, str]:
        """Return saved output paths for notebook display cells."""

        output_paths = {
            "raw_results": os.path.join(self.results_dir, "part2_raw_results.csv"),
            "aggregated_results": os.path.join(self.results_dir, "part2_aggregated_results.csv"),
            "ablation_plot": os.path.join(self.figures_dir, "part2_ablation_comparison.png"),
        }
        return output_paths

    def run(self) -> pd.DataFrame:
        """Run controlled baseline-vs-improved ablations for permuted images."""

        rows: List[dict] = []
        model_name = self.config.get("model_name", "resnet18")
        grid_sizes = [int(value) for value in self.config.get("grid_sizes", [2, 3, 4])]
        permutation_seed = int(self.config.get("permutation_seed", 42))
        eval_permutations = int(self.config.get("num_permutations", 2))

        for seed in self.config.get("seeds", [0]):
            seed_everything(int(seed), deterministic=bool(self.config.get("deterministic", False)))
            train_samples, val_samples, _ = self.load_data(seed=int(seed))
            for grid_size in grid_sizes:
                fixed_eval_permutations = [identity_permutation(grid_size)] + generate_permutations(
                    grid_size, eval_permutations, permutation_seed
                )
                training_permutation_pool = generate_permutations(
                    grid_size,
                    int(self.config.get("train_permutation_pool_size", max(4, eval_permutations))),
                    seed=permutation_seed + int(seed),
                )
                for ablation in _ablation_configs(self.config):
                    for permutation_id, permutation in enumerate(fixed_eval_permutations):
                        # Permutation augmentation samples from a pool during training,
                        # then evaluates on the same fixed permutations as the baseline.
                        train_loader, val_loader = build_dataloaders(
                            train_samples,
                            val_samples,
                            image_size=int(self.config.get("image_size", 224)),
                            grid_size=grid_size,
                            permutation=None if ablation.get("use_permutation_augmentation") else permutation,
                            random_permutations=training_permutation_pool if ablation.get("use_permutation_augmentation") else None,
                            seed=int(seed),
                            batch_size=int(self.config.get("batch_size", 32)),
                            num_workers=int(self.config.get("num_workers", 2)),
                            standard_augmentation=bool(ablation.get("use_standard_augmentation", False)),
                        )
                        _, val_loader = build_dataloaders(
                            train_samples,
                            val_samples,
                            image_size=int(self.config.get("image_size", 224)),
                            grid_size=grid_size,
                            permutation=permutation,
                            seed=int(seed),
                            batch_size=int(self.config.get("batch_size", 32)),
                            num_workers=int(self.config.get("num_workers", 2)),
                            standard_augmentation=False,
                        )
                        model = get_model(
                            model_name,
                            num_classes=int(self.config.get("num_classes", 2)),
                            pretrained=bool(ablation.get("use_pretrained", False)),
                            device=self.device,
                            freeze_backbone=bool(ablation.get("freeze_backbone", False)),
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
                                "part": "part2",
                                "run_id": self.run_id,
                                "config_name": self.config.get("config_name", "part2_improvement"),
                                "model_name": model_name,
                                "ablation_name": ablation["name"],
                                "use_pretrained": bool(ablation.get("use_pretrained", False)),
                                "use_standard_augmentation": bool(ablation.get("use_standard_augmentation", False)),
                                "use_permutation_augmentation": bool(ablation.get("use_permutation_augmentation", False)),
                                "freeze_backbone": bool(ablation.get("freeze_backbone", False)),
                                "grid_size": grid_size,
                                "num_tiles": grid_size * grid_size,
                                "permutation_id": permutation_id,
                                "permutation_seed": None if permutation_id == 0 else permutation_seed,
                                "seed": int(seed),
                                **metrics,
                            }
                        )
                        save_csv(rows, os.path.join(self.results_dir, "part2_raw_results.csv"))

        raw_results = pd.DataFrame(rows)
        aggregated = aggregate_accuracy(raw_results, ["ablation_name", "model_name", "grid_size", "num_tiles"])
        save_csv(aggregated, os.path.join(self.results_dir, "part2_aggregated_results.csv"))
        _plot_ablation_results(aggregated, os.path.join(self.figures_dir, "part2_ablation_comparison.png"))
        aggregated_results = aggregated
        return aggregated_results


def run_part2(config: Dict) -> pd.DataFrame:
    """Run controlled baseline-vs-improved ablations for permuted images."""

    experiment = Part2ImprovementExperiment(config)
    aggregated_results = experiment.run()
    return aggregated_results


def main(config_path: str) -> pd.DataFrame:
    """Run Part 2 from a YAML config path."""

    config = load_experiment_config(config_path)
    aggregated_results = run_part2(config)
    return aggregated_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    main(args.config)
