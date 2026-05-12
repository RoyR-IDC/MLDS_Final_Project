from PIL import Image
import pytest

from src.preprocessing.labels import AnimalLabel, parse_label_from_filename
from src.preprocessing.samples import class_counts, discover_samples, stratified_split


def _make_image(path):
    Image.new("RGB", (8, 8), (127, 127, 127)).save(path)


def test_parse_label_from_filename_returns_plain_ints():
    assert parse_label_from_filename("cat.123.jpg") == int(AnimalLabel.CAT)
    assert parse_label_from_filename("dog.456.jpg") == int(AnimalLabel.DOG)


def test_parse_label_from_filename_rejects_unknown_label():
    with pytest.raises(ValueError, match="Cannot parse"):
        parse_label_from_filename("bird.123.jpg")


def test_discover_samples_filters_unlabeled_images_and_counts_classes(tmp_path):
    _make_image(tmp_path / "cat.0.jpg")
    _make_image(tmp_path / "dog.1.jpg")
    _make_image(tmp_path / "unknown.2.jpg")

    samples = discover_samples(str(tmp_path))

    assert len(samples) == 2
    assert class_counts(samples) == {"cat": 1, "dog": 1}


def test_stratified_split_preserves_all_samples():
    samples = [(f"cat.{index}.jpg", 0) for index in range(4)] + [(f"dog.{index}.jpg", 1) for index in range(4)]

    train, val, test = stratified_split(samples, val_fraction=0.25, test_fraction=0.25, seed=123)

    assert len(train) == 4
    assert len(val) == 2
    assert len(test) == 2
    assert sorted(train + val + test) == sorted(samples)
