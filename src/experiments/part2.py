"""Part 2 ResNet-18 improvement ablation experiments."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
from torch.utils.data import DataLoader
from torch._C import device as TorchDevice

from src.evaluation.experiment_results import (
    get_device,
    load_experiment_samples,
    load_part1_model_baseline_aggregated,
    load_part1_model_baseline_raw_rows,
    plot_ablation_results,
)
from src.experiments.part1 import (
    _result_row_key,
    _run_training_with_progress_saves,
    build_pending_result_row,
    build_training_spec,
    get_executable_tile_permutation_records,
)
from src.experiments.results import experiment_output_paths, save_aggregated_accuracy, save_rows, save_run_rows
from src.preprocessing.augmentations import (
    BatchAugmentation,
    CompositeBatchAugmentation,
    RandomPatchShuffle,
    SameLabelCutMix,
)
from src.preprocessing.dataloaders import build_dataloaders
from src.preprocessing.samples import Sample
from src.preprocessing.tile_permutations import (
    TilePermutation,
    TilePermutationRecord,
    build_tile_permutation_records,
    random_tile_permutation,
)
from src.training.curriculum import CurriculumSchedule, TrainingStage
from src.utils.config import CVExperimentConfig


PATCH_SHUFFLE_DEFAULT_TILES = 3
CORRUPTION_PROBABILITY_SCHEDULE = [0.1, 0.3, 0.5, 0.8]


def _ablation_augmentation_name(ablation: Mapping[str, Any]) -> str:
    return str(ablation.get("augmentation", "none"))


def _ablation_curriculum_name(ablation: Mapping[str, Any]) -> str | None:
    curriculum = ablation.get("curriculum")
    return None if curriculum in (None, "", "none") else str(curriculum)


def _patch_shuffle_tiles(record: TilePermutationRecord) -> int:
    return int(record.tiles_per_side or PATCH_SHUFFLE_DEFAULT_TILES)


def _stage_epochs(total_epochs: int, num_stages: int) -> list[int]:
    if num_stages < 1:
        raise ValueError("num_stages must be at least 1")
    base = max(1, total_epochs // num_stages)
    epochs = [base for _ in range(num_stages)]
    for index in range(max(0, total_epochs - base * num_stages)):
        epochs[index % num_stages] += 1
    return epochs


def build_ablation_batch_augmentation(
    ablation: Mapping[str, Any],
    record: TilePermutationRecord,
) -> BatchAugmentation | None:
    """Build the batch-level augmentation strategy for one Part 2 ablation."""

    augmentation_name = _ablation_augmentation_name(ablation)
    if augmentation_name == "same_label_cutmix":
        return SameLabelCutMix()
    if augmentation_name == "patch_shuffle":
        return RandomPatchShuffle(tiles_per_side=_patch_shuffle_tiles(record))
    if augmentation_name == "combined_corruptions":
        return CompositeBatchAugmentation(
            [
                SameLabelCutMix(),
                RandomPatchShuffle(tiles_per_side=_patch_shuffle_tiles(record)),
            ]
        )
    return None


def build_ablation_dataloaders(
    *,
    ablation: Mapping[str, Any],
    record: TilePermutationRecord,
    train_samples: Sequence[Sample],
    validation_samples: Sequence[Sample],
    config: CVExperimentConfig,
) -> tuple[DataLoader, DataLoader]:
    """Build dataloaders for a non-curriculum Part 2 ablation."""

    augmentation_name = _ablation_augmentation_name(ablation)
    uses_patch_shuffle = augmentation_name in {"patch_shuffle", "combined_corruptions"}
    loader_tiles = _patch_shuffle_tiles(record) if uses_patch_shuffle else int(record.tiles_per_side or 1)
    image_augmentation = "random_erasing" if augmentation_name in {"random_erasing", "combined_corruptions"} else None
    return build_dataloaders(
        train_samples=train_samples,
        val_samples=validation_samples,
        image_size=config.image_size,
        tiles_per_side=loader_tiles,
        tile_permutation=record.tile_permutation,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        standard_augmentation=False,
        image_augmentation=image_augmentation,
    )


def _permutation_for_stage(
    *,
    tiles_per_side: int,
    record: TilePermutationRecord,
    config: CVExperimentConfig,
) -> TilePermutation:
    if record.tiles_per_side == tiles_per_side and record.tile_permutation is not None:
        return record.tile_permutation
    return random_tile_permutation(tiles_per_side, seed=int(config.seed) + int(record.tile_permutation_id))


def _difficulty_stage_tiles(record: TilePermutationRecord) -> list[int | None]:
    target = int(record.tiles_per_side or 1)
    if target <= 1:
        return [None]
    return [None, *[tiles for tiles in (2, 3, 4) if tiles <= target]]


def build_curriculum_schedule(
    *,
    ablation: Mapping[str, Any],
    record: TilePermutationRecord,
    train_samples: Sequence[Sample],
    config: CVExperimentConfig,
) -> CurriculumSchedule | None:
    """Build an optional curriculum schedule for one Part 2 ablation."""

    curriculum_name = _ablation_curriculum_name(ablation)
    if curriculum_name is None:
        return None

    if curriculum_name == "permutation_difficulty":
        stage_tiles = _difficulty_stage_tiles(record)
        epochs = _stage_epochs(int(config.epochs), len(stage_tiles))
        stages: list[TrainingStage] = []
        for stage_epochs, tiles_per_side in zip(epochs, stage_tiles):
            tile_permutation = (
                None
                if tiles_per_side is None
                else _permutation_for_stage(tiles_per_side=tiles_per_side, record=record, config=config)
            )
            train_loader, _ = build_dataloaders(
                train_samples=train_samples,
                val_samples=train_samples[: max(1, min(len(train_samples), int(config.batch_size)))],
                image_size=config.image_size,
                tiles_per_side=tiles_per_side or 1,
                tile_permutation=tile_permutation,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                standard_augmentation=False,
            )
            stage_name = "original" if tiles_per_side is None else f"{tiles_per_side}x{tiles_per_side}_permutation"
            stages.append(
                TrainingStage(
                    name=stage_name,
                    epochs=stage_epochs,
                    train_loader=train_loader,
                    metadata={"tiles_per_side": tiles_per_side},
                )
            )
        return CurriculumSchedule(stages)

    if curriculum_name == "corruption_probability":
        target_tiles = max(int(record.tiles_per_side or PATCH_SHUFFLE_DEFAULT_TILES), PATCH_SHUFFLE_DEFAULT_TILES)
        tile_permutation = record.tile_permutation or _permutation_for_stage(
            tiles_per_side=target_tiles,
            record=record,
            config=config,
        )
        epochs = _stage_epochs(int(config.epochs), len(CORRUPTION_PROBABILITY_SCHEDULE))
        stages = []
        for stage_epochs, probability in zip(epochs, CORRUPTION_PROBABILITY_SCHEDULE):
            train_loader, _ = build_dataloaders(
                train_samples=train_samples,
                val_samples=train_samples[: max(1, min(len(train_samples), int(config.batch_size)))],
                image_size=config.image_size,
                tiles_per_side=target_tiles,
                tile_permutation=tile_permutation,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                standard_augmentation=False,
                tile_permutation_probability=probability,
            )
            stages.append(
                TrainingStage(
                    name=f"permuted_probability_{probability:.1f}",
                    epochs=stage_epochs,
                    train_loader=train_loader,
                    metadata={"tile_permutation_probability": probability},
                )
            )
        return CurriculumSchedule(stages)

    raise ValueError(f"Unsupported curriculum: {curriculum_name}")


def collect_part2_ablation_results(
    *,
    config: CVExperimentConfig,
    ablations: Sequence[Mapping[str, Any]],
    train_samples: Sequence[tuple[str, int]],
    validation_samples: Sequence[tuple[str, int]],
    tile_permutation_records: Sequence[TilePermutationRecord],
    device: TorchDevice,
    run_id: str,
    raw_results_output_path: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Train all Part 2 ablations across tile-permutation records."""

    model_name = getattr(config, "model_name", config.model_names[0])
    executable_records = get_executable_tile_permutation_records(tile_permutation_records)
    rows: list[dict[str, Any]] = []
    row_indices: dict[tuple[Any, ...], int] = {}

    for ablation in ablations:
        for record in executable_records:
            row = build_pending_result_row(
                config=config,
                run_id=run_id,
                model_name=model_name,
                record=record,
                seed=config.seed,
                ablation_name=str(ablation["name"]),
            )
            row_indices[_result_row_key(row)] = len(rows)
            rows.append(row)

    if raw_results_output_path:
        save_run_rows(rows=rows, output_path=raw_results_output_path, run_id=run_id, model_name=model_name)
        print(f"Saved {len(rows)} pending row(s) for model '{model_name}' to {raw_results_output_path}.")

    for ablation in ablations:
        print()
        print("=" * 80)
        print(f"Running ablation: {ablation['name']}")

        for record_index, record in enumerate(executable_records, start=1):
            print()
            print(
                f"[{record_index}/{len(executable_records)}] model={model_name}, ablation={ablation['name']}, "
                f"tiles_per_side={record.tiles_per_side}, tile_permutation_id={record.tile_permutation_id}, "
                f"seed={record.tile_permutation_seed}"
            )
            print("Building dataloaders...")
            train_loader, validation_loader = build_ablation_dataloaders(
                ablation=ablation,
                record=record,
                train_samples=train_samples,
                validation_samples=validation_samples,
                config=config,
            )
            batch_augmentation = build_ablation_batch_augmentation(ablation, record)
            curriculum_schedule = build_curriculum_schedule(
                ablation=ablation,
                record=record,
                train_samples=train_samples,
                config=config,
            )
            print(f"Built dataloaders: {len(train_loader)} train batches, {len(validation_loader)} validation batches.")
            tiles_label = record.tiles_per_side or 1
            progress_desc = (
                f"{model_name} {ablation['name']} "
                f"{tiles_label}x{tiles_label} permutation {record.tile_permutation_id}"
            )
            spec = build_training_spec(
                config=config,
                model_name=model_name,
                train_loader=train_loader,
                val_loader=validation_loader,
                device=device,
                run_id=run_id,
                record=record,
                seed=config.seed,
                overrides={
                    "pretrained": bool(ablation.get("use_pretrained", True)),
                },
                ablation_name=str(ablation["name"]),
                progress_desc=progress_desc,
                batch_augmentation=batch_augmentation,
                curriculum_schedule=curriculum_schedule,
                metadata_overrides={
                    "augmentation_name": _ablation_augmentation_name(ablation),
                    "batch_augmentation_name": getattr(batch_augmentation, "name", None),
                    "curriculum_name": _ablation_curriculum_name(ablation),
                    "curriculum_stages": (
                        curriculum_schedule.stage_names if curriculum_schedule is not None else None
                    ),
                },
            )
            checkpoint_config = getattr(spec, "checkpoint_config", None)
            print(f"Best checkpoint path: {getattr(checkpoint_config, 'best_path', None)}")
            print(f"Raw results path: {raw_results_output_path}")
            row_key = (str(ablation["name"]), record.tiles_per_side or 0, record.tile_permutation_id)
            _run_training_with_progress_saves(
                spec=spec,
                config=config,
                run_id=run_id,
                record=record,
                seed=config.seed,
                rows=rows,
                row_index=row_indices[row_key],
                raw_results_output_path=raw_results_output_path,
                ablation_name=str(ablation["name"]),
            )
    return rows


def run_part2_improvement_experiments(
    config: CVExperimentConfig,
    device: Optional[TorchDevice] = None,
) -> pd.DataFrame:
    """Run Part 2 ablations, save raw/aggregated results, and write the comparison plot."""

    model_name = getattr(config, "model_name", config.model_names[0])
    resolved_device = device or get_device(config)
    train_samples, validation_samples, _ = load_experiment_samples(config, seed=config.seed)
    tile_permutation_records = build_tile_permutation_records(
        tiles_per_side_values=config.tiles_per_side_values,
        num_tile_permutations=config.num_tile_permutations,
        seed=config.seed,
        include_baseline=True,
    )
    run_id = datetime.now(timezone.utc).strftime("part2_%Y%m%d_%H%M%S")
    output_paths = experiment_output_paths(config.results_dir, config.figures_dir, config.part)

    rows = load_part1_model_baseline_raw_rows(config, model_name)
    rows.extend(
        collect_part2_ablation_results(
            config=config,
            ablations=getattr(config, "ablations"),
            train_samples=train_samples,
            validation_samples=validation_samples,
            tile_permutation_records=tile_permutation_records,
            device=resolved_device,
            run_id=run_id,
            raw_results_output_path=output_paths["raw_results"],
        )
    )

    save_rows(rows, output_paths["raw_results"])
    raw_results = pd.DataFrame(rows)
    aggregated_results = save_aggregated_accuracy(
        raw_results,
        group_columns=["model_name", "ablation_name", "tiles_per_side", "num_tiles"],
        output_path=output_paths["aggregated_results"],
    )

    part1_baseline_aggregated = load_part1_model_baseline_aggregated(config, model_name)
    has_regular_baseline = (
        "ablation_name" in aggregated_results.columns
        and (aggregated_results["ablation_name"] == "regular_part1").any()
    )
    if not has_regular_baseline and not part1_baseline_aggregated.empty:
        aggregated_results = pd.concat([part1_baseline_aggregated, aggregated_results], ignore_index=True, sort=False)
        aggregated_results.to_csv(output_paths["aggregated_results"], index=False)

    plot_ablation_results(aggregated_results, output_paths["figure"])
    return aggregated_results
