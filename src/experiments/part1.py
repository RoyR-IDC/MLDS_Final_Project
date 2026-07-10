"""Part 1 model and tile-permutation experiments."""

from __future__ import annotations

import importlib
import os
from time import perf_counter
from typing import Any, Optional, Sequence

import pandas as pd
import torch
from torch._C import device as TorchDevice
from torch.utils.data import DataLoader

from src.evaluation.experiment_results import get_device, load_experiment_samples, plot_accuracy_vs_tiles
from src.experiments.results import (
    aggregate_accuracy,
    build_result_row,
    experiment_intermediate_figure_path,
    experiment_output_paths,
    save_aggregated_accuracy,
    save_rows,
    save_run_rows,
)
import src.experiments.training_runs as _training_runs

# Notebook kernels often keep imported dependencies alive across saved source edits.
# Refresh the shared training helpers before binding their functions below.
_training_runs = importlib.reload(_training_runs)

from src.experiments.training_runs import (
    build_pending_training_result_row,
    build_experiment_run_id,
    build_training_run_spec,
    checkpoint_dir_path,
    checkpoints_enabled,
    format_dataloader_summary,
    format_elapsed_time,
    format_stage_summary,
    train_model_and_save_progress,
    training_result_row_key,
)
from src.models.factory import get_imagenet_pretrained_model
from src.models.registry import TIMM_MODEL_IDS
from src.preprocessing.dataloaders import build_dataloaders
from src.preprocessing.labels import AnimalLabel
from src.preprocessing.image_transforms import make_tile_compatible_image_size
from src.preprocessing.samples import Sample
from src.preprocessing.tile_permutations import TilePermutationRecord, build_tile_permutation_records
from src.training.run import TrainingResult
from src.utils.config import CVExperimentConfig


UNFROZEN_PRETRAINED_BINARY_HEAD_ABLATION = "unfrozen_pretrained_binary_head"
ZERO_SHOT_FULL_PRETRAINED_HEAD_ABLATION = "zero_shot_full_pretrained_head"
IMAGENET1K_DOG_CLASS_INDICES = tuple(range(151, 269))
IMAGENET1K_DOMESTIC_CAT_CLASS_INDICES = tuple(range(281, 286))
IMAGENET1K_NUM_CLASSES = 1000


def get_executable_tile_permutation_records(records: Sequence[TilePermutationRecord]) -> list[TilePermutationRecord]:
    """Return tile-permutation records that correspond to actual runs."""

    return list(records)


def completed_part1_variant_keys(raw_results_path: str, ablation_name: str) -> set[tuple[str, int | None, int]]:
    """Return completed Part 1 model/grid/permutation keys for one variant."""

    if not os.path.exists(raw_results_path):
        return set()
    raw = pd.read_csv(raw_results_path)
    if "ablation_name" not in raw.columns:
        return set()
    subset = raw[raw["ablation_name"].astype(str) == str(ablation_name)]
    if "run_status" in subset.columns:
        subset = subset[subset["run_status"].astype(str) == "completed"]
    return {
        (
            str(row.model_name),
            None if pd.isna(row.tiles_per_side) else int(row.tiles_per_side),
            int(row.tile_permutation_id),
        )
        for row in subset.itertuples(index=False)
    }


def missing_records_for_part1_variant(
    *,
    raw_results_path: str,
    ablation_name: str,
    model_name: str,
    records: Sequence[TilePermutationRecord],
) -> list[TilePermutationRecord]:
    """Return records that still need a completed row for a Part 1 variant."""

    completed = completed_part1_variant_keys(raw_results_path, ablation_name)
    missing = []
    for record in records:
        key = (
            str(model_name),
            None if record.tiles_per_side is None else int(record.tiles_per_side),
            int(record.tile_permutation_id),
        )
        if key not in completed:
            missing.append(record)
    return missing


def initialize_tile_permutation_result_rows(
    *,
    config: CVExperimentConfig,
    run_id: str,
    model_name: str,
    records: Sequence[TilePermutationRecord],
    seed: int,
    ablation_name: str | None = None,
) -> tuple[list[dict[str, Any]], dict[tuple[Any, ...], int]]:
    """Create pending result rows and their update indexes for a model."""

    rows = [
        build_pending_training_result_row(
            config=config,
            run_id=run_id,
            model_name=model_name,
            record=record,
            seed=seed,
            ablation_name=ablation_name,
        )
        for record in records
    ]
    row_indices = {training_result_row_key(row): index for index, row in enumerate(rows)}
    return rows, row_indices


def build_tile_permutation_dataloaders(
    *,
    config: CVExperimentConfig,
    model_name: str,
    train_samples: Sequence[Sample],
    validation_samples: Sequence[Sample],
    record: TilePermutationRecord,
) -> tuple[DataLoader, DataLoader]:
    """Build dataloaders for one Part 1 tile-permutation run."""

    fixed_input_size = config.image_size if model_name in TIMM_MODEL_IDS else None
    return build_dataloaders(
        train_samples=train_samples,
        val_samples=validation_samples,
        image_size=config.image_size,
        tiles_per_side=record.tiles_per_side or 1,
        tile_permutation=record.tile_permutation,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        standard_augmentation=False,
        output_image_size=fixed_input_size,
    )


def _print_tile_permutation_run_header(
    *,
    record_index: int,
    num_records: int,
    model_name: str,
    record: TilePermutationRecord,
    stage_summary: str,
) -> None:
    print()
    print("=" * 80)
    print(
        f"[{record_index}/{num_records}]\nmodel={model_name}, "
        f"tiles_per_side={record.tiles_per_side}, tile_permutation_id={record.tile_permutation_id}, "
        f"tile_permutation_name={record.tile_permutation_name}, seed={record.tile_permutation_seed}\n{stage_summary}"
    )


def train_single_tile_permutation_run(
    *,
    config: CVExperimentConfig,
    model_name: str,
    run_id: str,
    train_samples: Sequence[Sample],
    validation_samples: Sequence[Sample],
    record: TilePermutationRecord,
    seed: int,
    device: TorchDevice,
    rows: list[dict[str, Any]],
    row_indices: dict[tuple[Any, ...], int],
    session_start_time: Optional[float] = None,
    raw_results_output_path: Optional[str] = None,
    ablation_name: str | None = None,
    training_overrides: Optional[dict[str, Any]] = None,
    metadata_overrides: Optional[dict[str, Any]] = None,
) -> None:
    """Train one model on one tile-permutation record and update result rows."""

    run_start_time = perf_counter()
    resolved_session_start_time = session_start_time if session_start_time is not None else run_start_time
    print("Building dataloaders...")
    train_loader, validation_loader = build_tile_permutation_dataloaders(
        config=config,
        model_name=model_name,
        train_samples=train_samples,
        validation_samples=validation_samples,
        record=record,
    )
    print(f"Built dataloaders: {format_dataloader_summary(train_loader, validation_loader)}.")

    tiles_label = record.tiles_per_side or 1
    expected_input_size = (
        config.image_size
        if model_name in TIMM_MODEL_IDS
        else make_tile_compatible_image_size(config.image_size, record.tiles_per_side or 1)
    )
    permutation_label = (
        f"{record.tile_permutation_name} permutation"
        if record.tile_permutation_name
        else f"permutation {record.tile_permutation_id}"
    )
    spec = build_training_run_spec(
        config=config,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=validation_loader,
        device=device,
        run_id=run_id,
        record=record,
        seed=seed,
        overrides={"pretrained": config.pretrained, **dict(training_overrides or {})},
        ablation_name=ablation_name,
        metadata_overrides=metadata_overrides,
        progress_desc=f"{model_name} {tiles_label}x{tiles_label} {permutation_label}",
        expected_input_size=expected_input_size,
    )
    train_model_and_save_progress(
        spec=spec,
        config=config,
        run_id=run_id,
        record=record,
        seed=seed,
        rows=rows,
        row_index=row_indices[
            (
                (str(ablation_name), record.tiles_per_side or 0, record.tile_permutation_id)
                if ablation_name is not None
                else (record.tiles_per_side or 0, record.tile_permutation_id)
            )
        ],
        raw_results_output_path=raw_results_output_path,
        ablation_name=ablation_name,
    )
    print(f"Current run runtime: {format_elapsed_time(perf_counter() - run_start_time)}")
    print(f"Total training runtime: {format_elapsed_time(perf_counter() - resolved_session_start_time)}")


def train_model_on_tile_permutation_records(
    *,
    config: CVExperimentConfig,
    model_name: str,
    run_id: str,
    train_samples: Sequence[Sample],
    validation_samples: Sequence[Sample],
    tile_permutation_records: Sequence[TilePermutationRecord],
    seed: int,
    device: TorchDevice,
    session_start_time: Optional[float] = None,
    raw_results_output_path: Optional[str] = None,
    intermediate_figure_output_path: Optional[str] = None,
    ablation_name: str | None = None,
    training_overrides: Optional[dict[str, Any]] = None,
    metadata_overrides: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Train one model across tile-permutation records and collect result rows."""

    resolved_session_start_time = session_start_time if session_start_time is not None else perf_counter()
    executable_records = get_executable_tile_permutation_records(tile_permutation_records)
    rows, row_indices = initialize_tile_permutation_result_rows(
        config=config,
        run_id=run_id,
        model_name=model_name,
        records=executable_records,
        seed=seed,
        ablation_name=ablation_name,
    )

    if raw_results_output_path:
        save_run_rows(rows=rows, output_path=raw_results_output_path, run_id=run_id, model_name=model_name)
        print(f"Saved {len(rows)} pending row(s) for model '{model_name}'.")

    for record_index, record in enumerate(executable_records, start=1):
        stage_summary = format_stage_summary([("standard", int(config.epochs))])
        _print_tile_permutation_run_header(
            record_index=record_index,
            num_records=len(executable_records),
            model_name=model_name,
            record=record,
            stage_summary=stage_summary,
        )
        train_single_tile_permutation_run(
            config=config,
            model_name=model_name,
            run_id=run_id,
            train_samples=train_samples,
            validation_samples=validation_samples,
            record=record,
            seed=seed,
            device=device,
            rows=rows,
            row_indices=row_indices,
            raw_results_output_path=raw_results_output_path,
            session_start_time=resolved_session_start_time,
            ablation_name=ablation_name,
            training_overrides=training_overrides,
            metadata_overrides=metadata_overrides,
        )
    if intermediate_figure_output_path:
        raw_model_results = pd.DataFrame(rows)
        aggregated_model_results = aggregate_accuracy(
            raw_model_results,
            group_columns=["model_name", "tiles_per_side", "num_tiles"],
        )
        plot_accuracy_vs_tiles(
            aggregated_model_results,
            intermediate_figure_output_path,
            raw_results=raw_model_results,
            title=f"Intermediate Model Plot: Validation Accuracy by Tiling Level - {model_name}",
        )
        print(f"Saved intermediate model plot: {intermediate_figure_output_path}")
    return rows


def train_unfrozen_pretrained_binary_head_on_tile_permutation_records(
    *,
    config: CVExperimentConfig,
    model_name: str,
    run_id: str,
    train_samples: Sequence[Sample],
    validation_samples: Sequence[Sample],
    tile_permutation_records: Sequence[TilePermutationRecord],
    seed: int,
    device: TorchDevice,
    session_start_time: Optional[float] = None,
    raw_results_output_path: Optional[str] = None,
    intermediate_figure_output_path: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Train the pretrained binary-head Part 1 model with all parameters unfrozen."""

    return train_model_on_tile_permutation_records(
        config=config,
        model_name=model_name,
        run_id=run_id,
        train_samples=train_samples,
        validation_samples=validation_samples,
        tile_permutation_records=tile_permutation_records,
        seed=seed,
        device=device,
        session_start_time=session_start_time,
        raw_results_output_path=raw_results_output_path,
        intermediate_figure_output_path=intermediate_figure_output_path,
        ablation_name=UNFROZEN_PRETRAINED_BINARY_HEAD_ABLATION,
        training_overrides={"pretrained": True, "freeze_backbone": False},
        metadata_overrides={
            "epochs": int(config.epochs),
            "classification_head": "binary_linear",
        },
    )


def zero_shot_cat_dog_predictions(logits: torch.Tensor) -> torch.Tensor:
    """Return Dogs-vs-Cats predictions from ImageNet-1k logits."""

    if logits.ndim != 2 or logits.shape[1] != IMAGENET1K_NUM_CLASSES:
        raise ValueError(
            "Expected ImageNet-1k logits with shape "
            f"(batch_size, {IMAGENET1K_NUM_CLASSES}); got {tuple(logits.shape)}"
        )
    probabilities = torch.softmax(logits, dim=1)
    dog_probability = probabilities[:, IMAGENET1K_DOG_CLASS_INDICES].sum(dim=1)
    cat_probability = probabilities[:, IMAGENET1K_DOMESTIC_CAT_CLASS_INDICES].sum(dim=1)
    return torch.where(
        dog_probability > cat_probability,
        torch.full_like(dog_probability, int(AnimalLabel.DOG), dtype=torch.long),
        torch.full_like(cat_probability, int(AnimalLabel.CAT), dtype=torch.long),
    )


def evaluate_zero_shot_full_pretrained_head(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: TorchDevice,
) -> dict[str, float]:
    """Evaluate a native ImageNet-1k classifier as a zero-shot Dogs-vs-Cats model."""

    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            predictions = zero_shot_cat_dog_predictions(logits)
            correct += int((predictions == targets).sum().item())
            total += int(targets.numel())
    accuracy = correct / max(1, total)
    return {"val_accuracy": accuracy, "best_val_accuracy": accuracy}


def build_zero_shot_full_pretrained_head_result_row(
    *,
    config: CVExperimentConfig,
    run_id: str,
    model_name: str,
    record: TilePermutationRecord,
    seed: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    """Build a completed zero-shot result row for one Part 1 record."""

    result = TrainingResult.pending(
        model_name=model_name,
        metadata={
            "pretrained": True,
            "freeze_backbone": True,
            "epochs": 0,
            "classification_head": "imagenet_full_head",
        },
    )
    result.val_accuracy = metrics["val_accuracy"]
    result.best_val_accuracy = metrics["best_val_accuracy"]
    result.mark_completed(0.0)
    return build_result_row(
        config=config,
        run_id=run_id,
        model_name=model_name,
        record=record,
        seed=seed,
        metrics=result.latest_metrics(),
        ablation_name=ZERO_SHOT_FULL_PRETRAINED_HEAD_ABLATION,
    ) | result.metadata


def evaluate_zero_shot_full_pretrained_head_on_tile_permutation_records(
    *,
    config: CVExperimentConfig,
    model_name: str,
    run_id: str,
    train_samples: Sequence[Sample],
    validation_samples: Sequence[Sample],
    tile_permutation_records: Sequence[TilePermutationRecord],
    seed: int,
    device: TorchDevice,
    raw_results_output_path: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Evaluate one native pretrained ImageNet head over the Part 1 matrix."""

    executable_records = get_executable_tile_permutation_records(tile_permutation_records)
    model = get_imagenet_pretrained_model(model_name, device=device, freeze_backbone=True)
    rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(executable_records, start=1):
        _print_tile_permutation_run_header(
            record_index=record_index,
            num_records=len(executable_records),
            model_name=model_name,
            record=record,
            stage_summary="stages=0\n\tzero-shot ImageNet-1k head evaluation",
        )
        _, validation_loader = build_tile_permutation_dataloaders(
            config=config,
            model_name=model_name,
            train_samples=train_samples,
            validation_samples=validation_samples,
            record=record,
        )
        metrics = evaluate_zero_shot_full_pretrained_head(
            model=model,
            dataloader=validation_loader,
            device=device,
        )
        rows.append(
            build_zero_shot_full_pretrained_head_result_row(
                config=config,
                run_id=run_id,
                model_name=model_name,
                record=record,
                seed=seed,
                metrics=metrics,
            )
        )
        if raw_results_output_path:
            save_run_rows(
                rows=rows,
                output_path=raw_results_output_path,
                run_id=run_id,
                model_name=model_name,
            )
    return rows


def run_part1_experiments(config: CVExperimentConfig, device: Optional[TorchDevice] = None) -> pd.DataFrame:
    """Run Part 1 model comparison experiments and save raw/aggregated outputs."""

    session_start_time = perf_counter()
    resolved_device = device or get_device(config)
    train_samples, validation_samples, _ = load_experiment_samples(config, seed=config.seed)
    tile_permutation_records = build_tile_permutation_records(
        tiles_per_side_values=config.tiles_per_side_values,
        num_tile_permutations=config.num_tile_permutations,
        seed=config.seed,
        include_baseline=True,
    )
    run_id = build_experiment_run_id(config)
    output_paths = experiment_output_paths(config.results_dir, config.figures_dir, config.part)

    print(f"Raw results path: {output_paths['raw_results']}")
    if checkpoints_enabled(config):
        print(f"Checkpoint directory: {checkpoint_dir_path(config=config, run_id=run_id)}")
    else:
        print("Checkpointing disabled outside Google Colab.")

    rows: list[dict[str, Any]] = []
    for model_name in config.model_names:
        rows.extend(
            train_model_on_tile_permutation_records(
                config=config,
                model_name=model_name,
                run_id=run_id,
                train_samples=train_samples,
                validation_samples=validation_samples,
                tile_permutation_records=tile_permutation_records,
                seed=config.seed,
                device=resolved_device,
                raw_results_output_path=output_paths["raw_results"],
                intermediate_figure_output_path=experiment_intermediate_figure_path(
                    config.figures_dir,
                    config.part,
                    f"accuracy_vs_tiles_{model_name}",
                ),
                session_start_time=session_start_time,
            )
        )

    save_rows(rows, output_paths["raw_results"])
    raw_results = pd.DataFrame(rows)
    aggregated_results = save_aggregated_accuracy(
        raw_results,
        group_columns=["model_name", "tiles_per_side", "num_tiles"],
        output_path=output_paths["aggregated_results"],
    )
    plot_accuracy_vs_tiles(aggregated_results, output_paths["figure"], raw_results=raw_results)
    return aggregated_results
