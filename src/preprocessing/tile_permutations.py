"""Dependency-light tile-permutation generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Callable, Iterable, Sequence, TypeAlias, TypeGuard, cast


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


TILE_PERMUTATION_NAMES = ("easy", "medium", "large")
TilePermutationFunction: TypeAlias = Callable[[TilesPerSide], TilePermutation]


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


def easy_tile_permutation(tiles_per_side: TilesPerSide) -> TilePermutation:
    """Return a low-disruption deterministic local-swap permutation."""

    if tiles_per_side < 1:
        raise ValueError("tiles_per_side must be at least 1")
    flat_order = list(range(tiles_per_side * tiles_per_side))
    if tiles_per_side == 1:
        return TilePermutation(tiles_per_side=tiles_per_side, order=flat_order_to_matrix(tiles_per_side, flat_order))

    swap_count = max(1, tiles_per_side // 4)
    for index in range(swap_count):
        row = min(tiles_per_side - 1, index * 2)
        col = (index * 3) % (tiles_per_side - 1)
        _swap_positions(flat_order, tiles_per_side, (row, col), (row, col + 1))
    return TilePermutation(tiles_per_side=tiles_per_side, order=flat_order_to_matrix(tiles_per_side, flat_order))


def medium_tile_permutation(tiles_per_side: TilesPerSide) -> TilePermutation:
    """Return a medium-disruption deterministic shift-and-swap permutation."""

    if tiles_per_side < 1:
        raise ValueError("tiles_per_side must be at least 1")
    if tiles_per_side == 1:
        return identity_tile_permutation(tiles_per_side)

    grid = [[row * tiles_per_side + col for col in range(tiles_per_side)] for row in range(tiles_per_side)]
    for row, values in enumerate(grid):
        shift = row % 3 + 1
        grid[row] = values[shift:] + values[:shift]
    for col in range(tiles_per_side):
        values = [grid[row][col] for row in range(tiles_per_side)]
        shift = col % 2 + 1
        values = values[shift:] + values[:shift]
        for row, value in enumerate(values):
            grid[row][col] = value
    flat_order = [value for row in grid for value in row]
    swap_count = max(2, tiles_per_side // 2)
    for index in range(swap_count):
        row = (index * 2 + 1) % tiles_per_side
        col = (index * 3 + 1) % tiles_per_side
        _swap_positions(
            flat_order,
            tiles_per_side,
            (row, col),
            ((row + 1) % tiles_per_side, (col + 1) % tiles_per_side),
        )
    return TilePermutation(tiles_per_side=tiles_per_side, order=flat_order_to_matrix(tiles_per_side, flat_order))


def large_tile_permutation(tiles_per_side: TilesPerSide) -> TilePermutation:
    """Return a high-disruption deterministic global permutation."""

    if tiles_per_side < 1:
        raise ValueError("tiles_per_side must be at least 1")
    if tiles_per_side == 1:
        return identity_tile_permutation(tiles_per_side)

    matrix = [
        [
            (
                (tiles_per_side - 1 - col + row) % tiles_per_side,
                (tiles_per_side - 1 - row + 2 * col) % tiles_per_side,
            )
            for col in range(tiles_per_side)
        ]
        for row in range(tiles_per_side)
    ]
    flat_order = [old_row * tiles_per_side + old_col for row in matrix for old_row, old_col in row]
    if sorted(flat_order) != list(range(tiles_per_side * tiles_per_side)):
        flat_order = list(reversed(range(tiles_per_side * tiles_per_side)))

    swap_count = max(tiles_per_side, 3)
    for index in range(swap_count):
        row = index % tiles_per_side
        col = (index * 2) % tiles_per_side
        _swap_positions(
            flat_order,
            tiles_per_side,
            (row, col),
            (tiles_per_side - 1 - row, tiles_per_side - 1 - col),
        )
    return TilePermutation(tiles_per_side=tiles_per_side, order=flat_order_to_matrix(tiles_per_side, flat_order))


TILE_PERMUTATION_FUNCTIONS: dict[str, TilePermutationFunction] = {
    "easy": easy_tile_permutation,
    "medium": medium_tile_permutation,
    "large": large_tile_permutation,
}


def deterministic_tile_permutation(tiles_per_side: TilesPerSide, name: str) -> TilePermutation:
    """Return one named deterministic tile permutation."""

    try:
        permutation_function = TILE_PERMUTATION_FUNCTIONS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported tile permutation name: {name}") from exc
    return permutation_function(tiles_per_side)


def random_tile_permutation(tiles_per_side: TilesPerSide, seed: int) -> TilePermutation:
    """Generate one seeded random tile permutation."""

    flat_order = list(range(tiles_per_side * tiles_per_side))
    random.Random(seed).shuffle(flat_order)
    return TilePermutation(tiles_per_side=tiles_per_side, order=flat_order_to_matrix(tiles_per_side, flat_order))


def generate_tile_permutations(
    tiles_per_side: TilesPerSide,
    n: int,
    seed: int = 0,
) -> list[TilePermutation]:
    """Generate reusable seeded random tile permutations."""

    if n < 0:
        raise ValueError("n must be non-negative")
    rng = random.Random(seed)
    tile_permutations: list[TilePermutation] = []
    for _ in range(n):
        flat_order = list(range(tiles_per_side * tiles_per_side))
        rng.shuffle(flat_order)
        tile_permutations.append(
            TilePermutation(tiles_per_side=tiles_per_side, order=flat_order_to_matrix(tiles_per_side, flat_order))
        )
    return tile_permutations


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
