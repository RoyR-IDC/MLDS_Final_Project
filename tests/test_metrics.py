from src.utils.metrics import average_displacement, adjacency_preservation, displacement_entropy


def test_metrics_identity():
    G = 3
    idp = list(range(G * G))
    assert average_displacement(idp, G) == 0.0
    assert adjacency_preservation(idp, G) == 1.0
    ent = displacement_entropy(idp, G)
    assert ent == 0.0


def test_metrics_nontrivial():
    G = 3
    # simple swap of first and last
    p = list(range(G * G))
    p[0], p[-1] = p[-1], p[0]
    ad = adjacency_preservation(p, G)
    assert 0.0 <= ad <= 1.0
    disp = average_displacement(p, G)
    assert disp > 0.0
    ent = displacement_entropy(p, G)
    assert ent >= 0.0
