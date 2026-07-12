"""Dependency-light tile-permutation generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Iterable, Sequence, TypeAlias, TypeGuard, cast


TileCoordinate: TypeAlias = tuple[int, int]
TilePermutationOrder: TypeAlias = list[list[TileCoordinate]]
TilesPerSide: TypeAlias = int
JsonTileCoordinate: TypeAlias = Sequence[int]
JsonTilePermutationOrder: TypeAlias = Sequence[Sequence[JsonTileCoordinate]]


@dataclass(frozen=True)
class TilePermutation:
    """A square tile permutation mapping output positions to source positions.

    ``order[new_row][new_col]`` is the ``(old_row, old_col)`` source tile copied
    into the output tile at ``(new_row, new_col)``.
    """

    tiles_per_side: TilesPerSide
    order: TilePermutationOrder

    def __post_init__(self) -> None:
        if self.tiles_per_side < 1:
            raise ValueError("tiles_per_side must be at least 1")
        if len(self.order) != self.tiles_per_side:
            raise ValueError(f"order must have {self.tiles_per_side} rows")

        seen: set[TileCoordinate] = set()
        for row_index, row in enumerate(self.order):
            if len(row) != self.tiles_per_side:
                raise ValueError(f"order row {row_index} must have {self.tiles_per_side} columns")
            for coordinate in row:
                if not _is_tile_coordinate(coordinate):
                    raise ValueError("order entries must be (row, col) integer coordinates")
                old_row, old_col = coordinate
                if not (0 <= old_row < self.tiles_per_side and 0 <= old_col < self.tiles_per_side):
                    raise ValueError(
                        "order coordinates must be within "
                        f"[0, {self.tiles_per_side - 1}], got {(old_row, old_col)}"
                    )
                if coordinate in seen:
                    raise ValueError(f"source tile {coordinate} appears more than once")
                seen.add(coordinate)

        expected = self.tiles_per_side * self.tiles_per_side
        if len(seen) != expected:
            raise ValueError(f"order must contain each of the {expected} source tiles exactly once")


@dataclass(frozen=True)
class TilePermutationRecord:
    """Metadata for one reusable tile permutation."""

    tiles_per_side: TilesPerSide | None
    tile_permutation_id: int
    tile_permutation_seed: int
    tile_permutation: TilePermutation | None
    tile_permutation_name: str = ""


TILE_PERMUTATION_NAMES = ("easy", "medium", "hard")
ENHANCED_TILE_PERMUTATION_NAMES = (
    "easy",
    "easy2",
    "medium",
    "medium2",
    "hard",
    "hard2",
)
ENHANCED_BASELINE_TILE_PERMUTATION_ID = 0
ENHANCED_TILE_PERMUTATION_IDS = {
    name: index for index, name in enumerate(ENHANCED_TILE_PERMUTATION_NAMES, start=1)
}


@dataclass(frozen=True)
class DifficultyPermutationPreset:
    """Relative parameters for one deterministic difficulty level."""

    swap_fraction: float
    max_swap_distance_fraction: float
    row_shift_fraction: float
    col_shift_fraction: float
    use_global_reversal: bool = False


DIFFICULTY_PERMUTATION_PRESETS: dict[str, DifficultyPermutationPreset] = {
    "easy": DifficultyPermutationPreset(
        swap_fraction=0.05,
        max_swap_distance_fraction=0.15,
        row_shift_fraction=0.0,
        col_shift_fraction=0.0,
    ),
    "medium": DifficultyPermutationPreset(
        swap_fraction=0.14,
        max_swap_distance_fraction=0.35,
        row_shift_fraction=0.20,
        col_shift_fraction=0.20,
    ),
    "hard": DifficultyPermutationPreset(
        swap_fraction=0.28,
        max_swap_distance_fraction=0.80,
        row_shift_fraction=0.45,
        col_shift_fraction=0.45,
        use_global_reversal=True,
    ),
}


def _is_tile_coordinate(value: object) -> TypeGuard[TileCoordinate]:
    if not isinstance(value, tuple) or len(value) != 2:
        return False
    row, col = value
    return isinstance(row, int) and isinstance(col, int)


def flat_order_to_matrix(tiles_per_side: TilesPerSide, flat_order: Iterable[int]) -> TilePermutationOrder:
    """Convert an output-position to source-index order into a coordinate matrix."""

    values = list(flat_order)
    expected = tiles_per_side * tiles_per_side
    if len(values) != expected:
        raise ValueError(f"flat_order length must be {expected}, got {len(values)}")
    if sorted(values) != list(range(expected)):
        raise ValueError("flat_order must contain each source tile index exactly once")
    return [[divmod(values[row * tiles_per_side + col], tiles_per_side) for col in range(tiles_per_side)] for row in range(tiles_per_side)]


def matrix_to_flat_order(tile_permutation: TilePermutation) -> list[int]:
    """Convert a tile permutation matrix into a row-major source-index order."""

    flat_order: list[int] = []
    for row in tile_permutation.order:
        for old_row, old_col in row:
            flat_order.append(old_row * tile_permutation.tiles_per_side + old_col)
    return flat_order


def tile_permutation_to_jsonable(tile_permutation: TilePermutation | None) -> list[list[list[int]]] | None:
    """Return a JSON-serializable representation of a tile permutation."""

    if tile_permutation is None:
        return None
    return [[[old_row, old_col] for old_row, old_col in row] for row in tile_permutation.order]


def tile_permutation_from_jsonable(value: Any, tiles_per_side: TilesPerSide | None = None) -> TilePermutation | None:
    """Build a tile permutation from a JSON-loaded value."""

    if value is None or _is_nan(value):
        return None

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("tile_permutation must be a sequence, null, or NaN")

    values = cast(Sequence[Any], value)
    if tiles_per_side is None:
        tiles_per_side = len(values)
    if values and isinstance(values[0], int):
        flat_order = cast(Sequence[int], values)
        return TilePermutation(
            tiles_per_side=int(tiles_per_side),
            order=flat_order_to_matrix(int(tiles_per_side), flat_order),
        )

    json_order = cast(JsonTilePermutationOrder, values)
    order = [[(int(coordinate[0]), int(coordinate[1])) for coordinate in row] for row in json_order]
    return TilePermutation(tiles_per_side=int(tiles_per_side), order=order)


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def identity_tile_permutation(tiles_per_side: TilesPerSide) -> TilePermutation:
    """Return the identity tile permutation for a square tile layout."""

    if tiles_per_side < 1:
        raise ValueError("tiles_per_side must be at least 1")
    order = [[(row, col) for col in range(tiles_per_side)] for row in range(tiles_per_side)]
    return TilePermutation(tiles_per_side=tiles_per_side, order=order)


def _swap_positions(flat_order: list[int], tiles_per_side: TilesPerSide, first: TileCoordinate, second: TileCoordinate) -> None:
    first_index = first[0] * tiles_per_side + first[1]
    second_index = second[0] * tiles_per_side + second[1]
    flat_order[first_index], flat_order[second_index] = flat_order[second_index], flat_order[first_index]


def _validate_difficulty_fraction(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")


def _shift_row(values: list[int], shift: int) -> list[int]:
    if not values:
        return values
    shift = shift % len(values)
    if shift == 0:
        return values
    return values[shift:] + values[:shift]


def _matrix_to_flat_values(matrix: list[list[int]]) -> list[int]:
    return [value for row in matrix for value in row]


def _apply_row_and_column_shifts(
    flat_order: list[int],
    tiles_per_side: TilesPerSide,
    *,
    row_shift_fraction: float,
    col_shift_fraction: float,
) -> list[int]:
    row_shift = int(round((tiles_per_side - 1) * row_shift_fraction))
    col_shift = int(round((tiles_per_side - 1) * col_shift_fraction))
    if row_shift == 0 and col_shift == 0:
        return flat_order

    matrix = [
        flat_order[row * tiles_per_side : (row + 1) * tiles_per_side]
        for row in range(tiles_per_side)
    ]
    if row_shift:
        for row, values in enumerate(matrix):
            shift = (row_shift + row % max(1, row_shift)) % tiles_per_side
            matrix[row] = _shift_row(values, shift)
    if col_shift:
        for col in range(tiles_per_side):
            values = [matrix[row][col] for row in range(tiles_per_side)]
            shift = (col_shift + col % max(1, col_shift)) % tiles_per_side
            shifted = _shift_row(values, shift)
            for row, value in enumerate(shifted):
                matrix[row][col] = value
    return _matrix_to_flat_values(matrix)


def _difficulty_swap_count(num_tiles: int, swap_fraction: float) -> int:
    if swap_fraction == 0.0:
        return 0
    return min(num_tiles // 2, max(1, int(round(num_tiles * swap_fraction))))


def _deterministic_swap_pair(
    *,
    index: int,
    tiles_per_side: TilesPerSide,
    max_distance: int,
) -> tuple[TileCoordinate, TileCoordinate]:
    row = (index * 3 + index // 2) % tiles_per_side
    col = (index * 5 + index // 3) % tiles_per_side
    row_delta = ((index * 2 + 1) % (2 * max_distance + 1)) - max_distance
    col_delta = ((index * 3 + 2) % (2 * max_distance + 1)) - max_distance
    if row_delta == 0 and col_delta == 0:
        col_delta = 1
    row_delta = max(-max_distance, min(max_distance, row_delta))
    col_delta = max(-max_distance, min(max_distance, col_delta))
    other_row = (row + row_delta) % tiles_per_side
    other_col = (col + col_delta) % tiles_per_side
    if (other_row, other_col) == (row, col):
        other_col = (col + 1) % tiles_per_side
    return (row, col), (other_row, other_col)


def build_difficulty_tile_permutation(
    tiles_per_side: TilesPerSide,
    *,
    swap_fraction: float,
    max_swap_distance_fraction: float,
    row_shift_fraction: float,
    col_shift_fraction: float,
    use_global_reversal: bool = False,
) -> TilePermutation:
    """Build one deterministic tile permutation from relative difficulty parameters."""

    if tiles_per_side < 1:
        raise ValueError("tiles_per_side must be at least 1")
    _validate_difficulty_fraction(swap_fraction, "swap_fraction")
    _validate_difficulty_fraction(max_swap_distance_fraction, "max_swap_distance_fraction")
    _validate_difficulty_fraction(row_shift_fraction, "row_shift_fraction")
    _validate_difficulty_fraction(col_shift_fraction, "col_shift_fraction")

    num_tiles = tiles_per_side * tiles_per_side
    flat_order = list(range(num_tiles))
    if tiles_per_side == 1:
        return TilePermutation(tiles_per_side=tiles_per_side, order=flat_order_to_matrix(tiles_per_side, flat_order))

    if use_global_reversal:
        flat_order.reverse()
    flat_order = _apply_row_and_column_shifts(
        flat_order,
        tiles_per_side,
        row_shift_fraction=row_shift_fraction,
        col_shift_fraction=col_shift_fraction,
    )
    swap_count = _difficulty_swap_count(num_tiles, swap_fraction)
    max_distance = max(1, int(round((tiles_per_side - 1) * max_swap_distance_fraction)))
    for index in range(swap_count):
        first, second = _deterministic_swap_pair(
            index=index,
            tiles_per_side=tiles_per_side,
            max_distance=max_distance,
        )
        _swap_positions(flat_order, tiles_per_side, first, second)
    return TilePermutation(tiles_per_side=tiles_per_side, order=flat_order_to_matrix(tiles_per_side, flat_order))


def deterministic_tile_permutation(tiles_per_side: TilesPerSide, name: str) -> TilePermutation:
    """Return one named deterministic tile permutation."""

    try:
        preset = DIFFICULTY_PERMUTATION_PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported tile permutation name: {name}") from exc
    return build_difficulty_tile_permutation(
        tiles_per_side,
        swap_fraction=preset.swap_fraction,
        max_swap_distance_fraction=preset.max_swap_distance_fraction,
        row_shift_fraction=preset.row_shift_fraction,
        col_shift_fraction=preset.col_shift_fraction,
        use_global_reversal=preset.use_global_reversal,
    )


def base_tile_permutation_name(name: str | None) -> str:
    """Return the base difficulty label for an enhanced permutation name."""

    if name is None:
        return "baseline"
    normalized = str(name).strip().lower()
    if normalized in {"", "baseline"}:
        return "baseline"
    for base_name in TILE_PERMUTATION_NAMES:
        if normalized == base_name or normalized in {f"{base_name}2", f"{base_name}3"}:
            return base_name
    return normalized


def _variant_index(name: str) -> int:
    base_name = base_tile_permutation_name(name)
    if base_name == name:
        return 1
    suffix = str(name).removeprefix(base_name)
    return int(suffix) if suffix else 1


def _seeded_variant_flat_order(
    tiles_per_side: TilesPerSide,
    *,
    preset: DifficultyPermutationPreset,
    seed: int,
    variant_index: int,
) -> list[int]:
    num_tiles = tiles_per_side * tiles_per_side
    flat_order = list(range(num_tiles))
    if preset.use_global_reversal:
        flat_order.reverse()

    rng = random.Random(f"{seed}:{tiles_per_side}:{variant_index}")
    row_shift_fraction = min(1.0, max(0.0, preset.row_shift_fraction + rng.uniform(-0.04, 0.04)))
    col_shift_fraction = min(1.0, max(0.0, preset.col_shift_fraction + rng.uniform(-0.04, 0.04)))
    flat_order = _apply_row_and_column_shifts(
        flat_order,
        tiles_per_side,
        row_shift_fraction=row_shift_fraction,
        col_shift_fraction=col_shift_fraction,
    )

    swap_count = _difficulty_swap_count(num_tiles, preset.swap_fraction)
    max_distance = max(1, int(round((tiles_per_side - 1) * preset.max_swap_distance_fraction)))
    positions = [(row, col) for row in range(tiles_per_side) for col in range(tiles_per_side)]
    for _ in range(swap_count):
        first = rng.choice(positions)
        candidates = [
            position
            for position in positions
            if position != first
            and max(abs(position[0] - first[0]), abs(position[1] - first[1])) <= max_distance
        ]
        second = rng.choice(candidates or [position for position in positions if position != first])
        _swap_positions(flat_order, tiles_per_side, first, second)
    return flat_order


def deterministic_enhanced_tile_permutation(
    tiles_per_side: TilesPerSide,
    name: str,
    *,
    seed: int = 42,
) -> TilePermutation:
    """Return a deterministic enhanced permutation, preserving existing base labels."""

    normalized = str(name).strip().lower()
    if normalized in TILE_PERMUTATION_NAMES:
        return deterministic_tile_permutation(tiles_per_side, normalized)
    if normalized not in ENHANCED_TILE_PERMUTATION_IDS:
        raise ValueError(f"Unsupported enhanced tile permutation name: {name}")
    if tiles_per_side < 1:
        raise ValueError("tiles_per_side must be at least 1")
    if tiles_per_side == 1:
        return identity_tile_permutation(tiles_per_side)

    base_name = base_tile_permutation_name(normalized)
    preset = DIFFICULTY_PERMUTATION_PRESETS[base_name]
    flat_order = _seeded_variant_flat_order(
        tiles_per_side,
        preset=preset,
        seed=seed,
        variant_index=_variant_index(normalized),
    )
    return TilePermutation(tiles_per_side=tiles_per_side, order=flat_order_to_matrix(tiles_per_side, flat_order))


def build_enhanced_tile_permutation_records(
    tiles_per_side_values: Iterable[TilesPerSide],
    *,
    seed: int = 42,
    include_baseline: bool = True,
) -> list[TilePermutationRecord]:
    """Build expanded two-seed subset records with one shared 1x1 baseline."""

    records: list[TilePermutationRecord] = []
    emitted_baseline = False
    for tiles_per_side in tiles_per_side_values:
        resolved_tiles_per_side = int(tiles_per_side)
        if resolved_tiles_per_side < 1:
            raise ValueError("tiles_per_side values must be at least 1")
        if resolved_tiles_per_side == 1:
            if include_baseline and not emitted_baseline:
                records.append(
                    TilePermutationRecord(
                        tiles_per_side=None,
                        tile_permutation_id=ENHANCED_BASELINE_TILE_PERMUTATION_ID,
                        tile_permutation_seed=seed,
                        tile_permutation=None,
                        tile_permutation_name="baseline",
                    )
                )
                emitted_baseline = True
            continue

        for permutation_name in ENHANCED_TILE_PERMUTATION_NAMES:
            records.append(
                TilePermutationRecord(
                    tiles_per_side=resolved_tiles_per_side,
                    tile_permutation_id=ENHANCED_TILE_PERMUTATION_IDS[permutation_name],
                    tile_permutation_seed=seed,
                    tile_permutation=deterministic_enhanced_tile_permutation(
                        resolved_tiles_per_side,
                        permutation_name,
                        seed=seed,
                    ),
                    tile_permutation_name=permutation_name,
                )
            )
    return records


def build_tile_permutation_records(
    tiles_per_side_values: Iterable[TilesPerSide],
    num_tile_permutations: int,
    seed: int = 42,
    include_baseline: bool = True,
) -> list[TilePermutationRecord]:
    """Build stable named tile-permutation records for experiment reuse."""

    records: list[TilePermutationRecord] = []
    permutation_names = list(TILE_PERMUTATION_NAMES[:num_tile_permutations])
    if num_tile_permutations < 0:
        raise ValueError("num_tile_permutations must be non-negative")
    if len(permutation_names) != num_tile_permutations:
        raise ValueError(
            f"num_tile_permutations must be between 0 and {len(TILE_PERMUTATION_NAMES)} "
            f"for deterministic named permutations, got {num_tile_permutations}"
        )

    for tiles_per_side in tiles_per_side_values:
        resolved_tiles_per_side = int(tiles_per_side)
        if resolved_tiles_per_side < 1:
            raise ValueError("tiles_per_side values must be at least 1")
        if resolved_tiles_per_side == 1:
            if not include_baseline:
                continue
            for offset, permutation_name in enumerate(permutation_names, start=1):
                records.append(
                    TilePermutationRecord(
                        tiles_per_side=None,
                        tile_permutation_id=offset,
                        tile_permutation_seed=seed,
                        tile_permutation=None,
                        tile_permutation_name=permutation_name,
                    )
                )
            continue
        for offset, permutation_name in enumerate(permutation_names, start=1):
            records.append(
                TilePermutationRecord(
                    tiles_per_side=resolved_tiles_per_side,
                    tile_permutation_id=offset,
                    tile_permutation_seed=seed,
                    tile_permutation=deterministic_tile_permutation(resolved_tiles_per_side, permutation_name),
                    tile_permutation_name=permutation_name,
                )
            )
    return records
