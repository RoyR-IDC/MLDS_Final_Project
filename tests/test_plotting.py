import numpy as np
from PIL import Image

from src.preprocessing.tile_permutations import TilePermutationRecord, deterministic_tile_permutation
from src.utils.plotting import plot_tile_permutation_samples


def test_plot_tile_permutation_samples_uses_cat_dog_columns_and_variant_rows(tmp_path):
    cat_path = tmp_path / "cat.0.jpg"
    dog_path = tmp_path / "dog.0.jpg"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(cat_path)
    Image.fromarray(np.full((8, 8, 3), 255, dtype=np.uint8)).save(dog_path)
    records = [
        TilePermutationRecord(
            tiles_per_side=2,
            tile_permutation_id=1,
            tile_permutation_seed=42,
            tile_permutation=deterministic_tile_permutation(2, "easy"),
            tile_permutation_name="easy",
        ),
        TilePermutationRecord(
            tiles_per_side=2,
            tile_permutation_id=2,
            tile_permutation_seed=42,
            tile_permutation=deterministic_tile_permutation(2, "hard"),
            tile_permutation_name="hard",
        ),
    ]

    fig = plot_tile_permutation_samples(
        samples=[(str(cat_path), 0), (str(dog_path), 1)],
        tile_permutation_records=records,
        image_size=8,
        max_records=2,
    )

    titles = [axis.get_title() for axis in fig.axes]
    assert len(fig.axes) == 6
    assert titles[:2] == ["Cat regular", "Dog regular"]
    assert any("hard" in title for title in titles)


def test_plot_tile_permutation_samples_can_show_easy_medium_hard_rows(tmp_path):
    cat_path = tmp_path / "cat.0.jpg"
    dog_path = tmp_path / "dog.0.jpg"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(cat_path)
    Image.fromarray(np.full((8, 8, 3), 255, dtype=np.uint8)).save(dog_path)
    records = [
        TilePermutationRecord(
            tiles_per_side=2,
            tile_permutation_id=index,
            tile_permutation_seed=42,
            tile_permutation=deterministic_tile_permutation(2, name),
            tile_permutation_name=name,
        )
        for index, name in enumerate(("easy", "medium", "hard"), start=1)
    ]

    fig = plot_tile_permutation_samples(
        samples=[(str(cat_path), 0), (str(dog_path), 1)],
        tile_permutation_records=records,
        image_size=8,
        max_records=3,
    )

    titles = [axis.get_title() for axis in fig.axes]
    assert len(fig.axes) == 8
    assert any("easy" in title for title in titles)
    assert any("medium" in title for title in titles)
    assert any("hard" in title for title in titles)
