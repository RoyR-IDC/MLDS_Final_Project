from src.preprocessing.permutations import generate_permutations, identity_permutation


def test_generate_permutations_length():
    permutations = generate_permutations(3, 5, seed=123)
    assert len(permutations) == 5
    assert all(len(permutation) == 9 for permutation in permutations)


def test_identity_permutation():
    permutation = identity_permutation(4)
    assert permutation == list(range(16))
