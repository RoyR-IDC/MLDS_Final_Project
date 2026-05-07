from src.preprocessing.permutations import build_permutation_records, generate_permutations, identity_permutation


def test_generate_permutations_length():
    permutations = generate_permutations(3, 5, seed=123)
    assert len(permutations) == 5
    assert all(len(permutation) == 9 for permutation in permutations)


def test_identity_permutation():
    permutation = identity_permutation(4)
    assert permutation == list(range(16))


def test_build_permutation_records_uses_unified_seed_for_identity_and_random_records():
    records = build_permutation_records([1, 2], num_permutations=1, seed=123)

    assert {record.permutation_seed for record in records} == {123}
    assert records[0].grid_size == 1
    assert records[0].permutation_id == 0
    assert records[0].permutation == [0]
