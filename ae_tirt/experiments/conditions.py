"""Experiment condition generation."""

from itertools import product

from ae_tirt.utils.correlation import sample_random_correlation_matrix


def generate_conditions(grid: dict):
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    return [dict(zip(keys, row)) for row in product(*vals)]


def generate_tirt_conditions(
    npersons_levels=None,
    npairs_levels=None,
    correlation_type_levels=None,
    weight_sign_levels=None,
):
    """Generate Study-1 style factorial conditions."""
    if npersons_levels is None:
        npersons_levels = [300, 500, 1000]
    if npairs_levels is None:
        npairs_levels = [15, 30]
    if correlation_type_levels is None:
        correlation_type_levels = ["independent", "correlated"]
    if weight_sign_levels is None:
        weight_sign_levels = [1.0, 0.5]

    ntraits = 5
    w_range = (0.0, 1.0)
    b_range = (-1, 1)
    correlation_matrix_5 = [
        [1.0, -0.21, -0.53, -0.25, 0.00],
        [-0.21, 1.0, 0.27, 0.00, 0.40],
        [-0.53, 0.27, 1.0, 0.24, 0.00],
        [-0.25, 0.00, 0.24, 1.0, 0.00],
        [0.00, 0.40, 0.00, 0.00, 1.0],
    ]

    def get_config(npairs):
        if npairs == 15:
            return {"nblocks_per_trait": 6, "nitems_per_block": 2}
        if npairs == 30:
            return {"nblocks_per_trait": 12, "nitems_per_block": 2}
        raise ValueError(f"npairs must be 15 or 30; got {npairs}")

    conditions_list = []
    for npersons in npersons_levels:
        for npairs in npairs_levels:
            config = get_config(npairs)
            for correlation_type in correlation_type_levels:
                for weight_sign in weight_sign_levels:
                    condition = {
                        "npersons": npersons,
                        "ntraits": ntraits,
                        "npairs": npairs,
                        "nblocks_per_trait": config["nblocks_per_trait"],
                        "nitems_per_block": config["nitems_per_block"],
                        "weight_sign": weight_sign,
                        "w_range": w_range,
                        "b_range": b_range,
                        "comb_blocks": "random",
                        "correlation_type": correlation_type,
                    }
                    if correlation_type == "correlated":
                        condition["correlation_matrix"] = correlation_matrix_5
                    conditions_list.append(condition)
    return conditions_list


def generate_tirt_conditions_study2(
    ntraits_levels=None,
    npersons: int = 500,
    nblocks_per_trait: int = 9,
    nitems_per_block: int = 2,
    weight_sign: float = 0.5,
    correlation_range: tuple = (-0.5, 0.5),
):
    """Generate Study-2 conditions with varying ntraits."""
    if ntraits_levels is None:
        ntraits_levels = [10, 20]

    w_range = (0.65, 0.95)
    b_range = (-1, 1)
    conditions_list = []

    for ntraits in ntraits_levels:
        npairs = int(ntraits * nblocks_per_trait / nitems_per_block)
        for correlation_type in ["independent", "correlated"]:
            cond = {
                "npersons": npersons,
                "ntraits": ntraits,
                "npairs": npairs,
                "nblocks_per_trait": nblocks_per_trait,
                "nitems_per_block": nitems_per_block,
                "weight_sign": weight_sign,
                "w_range": w_range,
                "b_range": b_range,
                "comb_blocks": "random",
                "correlation_type": correlation_type,
            }
            if correlation_type == "correlated":
                low, high = correlation_range
                cond["correlation_sampler"] = (
                    lambda _cond, _r, ntraits=ntraits, low=low, high=high: sample_random_correlation_matrix(
                        ntraits=ntraits, low=low, high=high
                    )
                )
            conditions_list.append(cond)
    return conditions_list
