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
    """Build stable tile-permutation records for experiment reuse.

    The baseline is represented by ``tile_permutation=None``.
    """

    records: list[TilePermutationRecord] = []
    if include_baseline:
        records.append(
            TilePermutationRecord(
                tiles_per_side=None,
                tile_permutation_id=0,
                tile_permutation_seed=seed,
                tile_permutation=None,
            )
        )

    for tiles_per_side in tiles_per_side_values:
        if tiles_per_side == 1:
            continue
        for offset, tile_permutation in enumerate(
            generate_tile_permutations(tiles_per_side, num_tile_permutations, seed),
            start=1,
        ):
            records.append(
                TilePermutationRecord(
                    tiles_per_side=tiles_per_side,
                    tile_permutation_id=offset,
                    tile_permutation_seed=seed,
                    tile_permutation=tile_permutation,
                )
            )
    return records
