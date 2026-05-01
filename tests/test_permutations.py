from src.utils.permutations import generate_permutations, identity_permutation


def test_generate_permutations_length():
    perms = generate_permutations(3, 5, seed=123)
    assert len(perms) == 5
    assert all(len(p) == 9 for p in perms)


def test_identity_permutation():
    p = identity_permutation(4)
    assert p == list(range(16))
