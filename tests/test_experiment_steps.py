from src.preprocessing.permutations import PermutationRecord
from src.training.experiment_steps import get_executable_permutation_records


def test_executable_permutation_records_skip_duplicate_one_by_one_permutations():
    records = [
        PermutationRecord(grid_size=1, permutation_id=0, permutation_seed=42, permutation=[0]),
        PermutationRecord(grid_size=1, permutation_id=1, permutation_seed=42, permutation=[0]),
        PermutationRecord(grid_size=2, permutation_id=0, permutation_seed=42, permutation=[0, 1, 2, 3]),
        PermutationRecord(grid_size=2, permutation_id=1, permutation_seed=42, permutation=[1, 0, 3, 2]),
    ]

    executable_records = get_executable_permutation_records(records)

    assert [(record.grid_size, record.permutation_id) for record in executable_records] == [
        (1, 0),
        (2, 0),
        (2, 1),
    ]
