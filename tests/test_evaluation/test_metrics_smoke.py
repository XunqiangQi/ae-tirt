import numpy as np

from ae_tirt.evaluation.metrics import evaluate_traits


def test_evaluate_traits_returns_expected_keys():
    true_traits = np.array([[0.1, -0.1], [0.2, -0.2], [0.3, -0.3]])
    est_traits = true_traits.copy()
    out = evaluate_traits(est_traits, true_traits)

    assert "overall" in out
    assert "per_trait" in out
    assert "trait_1" in out["per_trait"]
    assert "rmse" in out["overall"]
