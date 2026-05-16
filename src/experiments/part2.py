"""Part 2 ResNet-18 improvement ablation experiments."""

from __future__ import annotations

import importlib
from time import perf_counter
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
from src.experiments.results import experiment_output_paths, save_aggregated_accuracy, save_rows, save_run_rows
import src.experiments.training_runs as _training_runs

if not hasattr(_training_runs, "checkpoints_enabled"):
    importlib.reload(_training_runs)

from src.experiments.training_runs import (
    build_pending_training_result_row,
    build_experiment_run_id,
    build_training_run_spec,
    checkpoint_dir_path,
    checkpoints_enabled,
    format_dataloader_summary,
    format_elapsed_time,
    format_stage_dataloader_summary,
    format_stage_summary,
    train_model_and_save_progress,
    training_result_row_key,
)
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
from src.training.run import TrainingRunSpec
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
    if augmentation_name == "combined_augmentations":
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
    uses_patch_shuffle = augmentation_name in {"patch_shuffle", "combined_augmentations"}
    loader_tiles = _patch_shuffle_tiles(record) if uses_patch_shuffle else int(record.tiles_per_side or 1)
    image_augmentation = "random_erasing" if augmentation_name in {"random_erasing", "combined_augmentations"} else None
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


def _planned_stage_items(
    *,
    ablation: Mapping[str, Any],
    record: TilePermutationRecord,
    config: CVExperimentConfig,
) -> list[tuple[str, int]]:
    """Return stage names and epoch counts without building stage dataloaders."""

    curriculum_name = _ablation_curriculum_name(ablation)
    if curriculum_name is None:
        return [("standard", int(config.epochs))]

    if curriculum_name == "permutation_difficulty":
        stage_tiles = _difficulty_stage_tiles(record)
        epochs = _stage_epochs(int(config.epochs), len(stage_tiles))
        names = [
            "original" if tiles_per_side is None else f"{tiles_per_side}x{tiles_per_side}_permutation"
            for tiles_per_side in stage_tiles
        ]
        return list(zip(names, epochs))

    if curriculum_name == "corruption_probability":
        epochs = _stage_epochs(int(config.epochs), len(CORRUPTION_PROBABILITY_SCHEDULE))
        names = [f"permuted_probability_{probability:.1f}" for probability in CORRUPTION_PROBABILITY_SCHEDULE]
        return list(zip(names, epochs))

    raise ValueError(f"Unsupported curriculum: {curriculum_name}")


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


def initialize_ablation_result_rows(
    *,
    config: CVExperimentConfig,
    model_name: str,
    run_id: str,
    ablations: Sequence[Mapping[str, Any]],
    records: Sequence[TilePermutationRecord],
) -> tuple[list[dict[str, Any]], dict[tuple[Any, ...], int]]:
    """Create pending Part 2 ablation result rows and their update indexes."""

    rows: list[dict[str, Any]] = []
    row_indices: dict[tuple[Any, ...], int] = {}
    for ablation in ablations:
        for record in records:
            row = build_pending_training_result_row(
                config=config,
                run_id=run_id,
                model_name=model_name,
                record=record,
                seed=config.seed,
                ablation_name=str(ablation["name"]),
            )
            row_indices[training_result_row_key(row)] = len(rows)
            rows.append(row)
    return rows, row_indices


def _part2_metadata_overrides(
    *,
    ablation: Mapping[str, Any],
    batch_augmentation: BatchAugmentation | None,
    curriculum_schedule: CurriculumSchedule | None,
) -> dict[str, Any]:
    return {
        "augmentation_name": _ablation_augmentation_name(ablation),
        "batch_augmentation_name": getattr(batch_augmentation, "name", None),
        "curriculum_name": _ablation_curriculum_name(ablation),
        "curriculum_stages": curriculum_schedule.stage_names if curriculum_schedule is not None else None,
    }


def build_part2_training_run_spec(
    *,
    config: CVExperimentConfig,
    model_name: str,
    run_id: str,
    ablation: Mapping[str, Any],
    record: TilePermutationRecord,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: TorchDevice,
    batch_augmentation: BatchAugmentation | None,
    curriculum_schedule: CurriculumSchedule | None,
) -> TrainingRunSpec:
    """Build the training spec for one Part 2 ablation/tile run."""

    tiles_label = record.tiles_per_side or 1
    progress_desc = (
        f"{model_name} [{ablation['name']}] "
        f"{tiles_label}x{tiles_label} permutation #{record.tile_permutation_id}. epochs progress"
    )
    return build_training_run_spec(
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
        metadata_overrides=_part2_metadata_overrides(
            ablation=ablation,
            batch_augmentation=batch_augmentation,
            curriculum_schedule=curriculum_schedule,
        ),
    )


def _print_ablation_tile_run_header(
    *,
    record_index: int,
    num_records: int,
    model_name: str,
    ablation: Mapping[str, Any],
    record: TilePermutationRecord,
    stage_summary: str,
) -> None:
    print()
    print(
        f"[{record_index}/{num_records}]\nmodel={model_name}, ablation={ablation['name']}, "
        f"tiles_per_side={record.tiles_per_side}, tile_permutation_id={record.tile_permutation_id}, "
        f"seed={record.tile_permutation_seed}\n{stage_summary}"
    )


def train_single_ablation_tile_run(
    *,
    config: CVExperimentConfig,
    model_name: str,
    run_id: str,
    ablation: Mapping[str, Any],
    record: TilePermutationRecord,
    train_samples: Sequence[Sample],
    validation_samples: Sequence[Sample],
    device: TorchDevice,
    rows: list[dict[str, Any]],
    row_indices: dict[tuple[Any, ...], int],
    session_start_time: Optional[float] = None,
    raw_results_output_path: Optional[str] = None,
) -> None:
    """Train one Part 2 ablation on one tile-permutation record."""

    run_start_time = perf_counter()
    resolved_session_start_time = session_start_time if session_start_time is not None else run_start_time
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
    print(f"Built dataloaders: {format_dataloader_summary(train_loader, validation_loader)}.")
    if curriculum_schedule is not None:
        print(f"Stage dataloaders: {format_stage_dataloader_summary(curriculum_schedule.stages)}.")
    spec = build_part2_training_run_spec(
        config=config,
        model_name=model_name,
        run_id=run_id,
        ablation=ablation,
        record=record,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        batch_augmentation=batch_augmentation,
        curriculum_schedule=curriculum_schedule,
    )
    row_key = (str(ablation["name"]), record.tiles_per_side or 0, record.tile_permutation_id)
    train_model_and_save_progress(
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
    print(f"Current run runtime: {format_elapsed_time(perf_counter() - run_start_time)}")
    print(f"Total training runtime: {format_elapsed_time(perf_counter() - resolved_session_start_time)}")


def train_part2_ablation_experiments(
    *,
    config: CVExperimentConfig,
    ablations: Sequence[Mapping[str, Any]],
    train_samples: Sequence[Sample],
    validation_samples: Sequence[Sample],
    tile_permutation_records: Sequence[TilePermutationRecord],
    device: TorchDevice,
    run_id: str,
    session_start_time: Optional[float] = None,
    raw_results_output_path: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Train all Part 2 ablations across tile-permutation records."""

    resolved_session_start_time = session_start_time if session_start_time is not None else perf_counter()
    model_name = getattr(config, "model_name", config.model_names[0])
    executable_records = list(tile_permutation_records)
    rows, row_indices = initialize_ablation_result_rows(
        config=config,
        model_name=model_name,
        run_id=run_id,
        ablations=ablations,
        records=executable_records,
    )

    if raw_results_output_path:
        save_run_rows(rows=rows, output_path=raw_results_output_path, run_id=run_id, model_name=model_name)
        print(f"Saved {len(rows)} pending row(s) for model '{model_name}'.")

    for ablation in ablations:
        print()
        print("=" * 80)
        print(f"Running ablation: {ablation['name']}")

        for record_index, record in enumerate(executable_records, start=1):
            stage_summary = format_stage_summary(
                _planned_stage_items(ablation=ablation, record=record, config=config)
            )
            _print_ablation_tile_run_header(
                record_index=record_index,
                num_records=len(executable_records),
                model_name=model_name,
                ablation=ablation,
                record=record,
                stage_summary=stage_summary,
            )
            train_single_ablation_tile_run(
                config=config,
                model_name=model_name,
                run_id=run_id,
                ablation=ablation,
                record=record,
                train_samples=train_samples,
                validation_samples=validation_samples,
                device=device,
                rows=rows,
                row_indices=row_indices,
                session_start_time=resolved_session_start_time,
                raw_results_output_path=raw_results_output_path,
            )
    return rows


def run_part2_improvement_experiments(
    config: CVExperimentConfig,
    device: Optional[TorchDevice] = None,
) -> pd.DataFrame:
    """Run Part 2 ablations, save raw/aggregated results, and write the comparison plot."""

    session_start_time = perf_counter()
    model_name = getattr(config, "model_name", config.model_names[0])
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

    rows = load_part1_model_baseline_raw_rows(config, model_name)
    rows.extend(
        train_part2_ablation_experiments(
            config=config,
            ablations=getattr(config, "ablations"),
            train_samples=train_samples,
            validation_samples=validation_samples,
            tile_permutation_records=tile_permutation_records,
            device=resolved_device,
            run_id=run_id,
            session_start_time=session_start_time,
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
