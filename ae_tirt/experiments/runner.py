"""Batch experiment runner."""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import torch

from ae_tirt.data.simulator import Sim_data_TIRT
from ae_tirt.evaluation.validators import evaluate_item_parameters, evaluate_model
from ae_tirt.models.ae_tirt import AE_TIRT
from ae_tirt.training.trainer import train_model
from ae_tirt.utils.paths import ensure_dir, get_safe_path

from .results import save_item_parameter_results, save_results


def run_experiments(conditions, run_fn):
    results = []
    for cond in conditions:
        results.append(run_fn(cond))
    return results


def _resolve_batch_size(nitems_per_block: int, batch_size: Optional[int]) -> int:
    """Resolve batch size from block size unless manually overridden."""
    if batch_size is not None:
        return int(batch_size)
    if nitems_per_block == 3:
        return 32
    if nitems_per_block == 2:
        return 16
    return 16


def batch_simulate_and_train(
    conditions_list,
    repeat=1,
    sim_data_root="Sim_data_result",
    num_epochs=500,
    batch_size=None,
    learning_rate=0.001,
    early_stopping_patience=20,
    penalty_weight_factor=0.1,
    weight_constraint: str = "standardized",
    link_function: str = "probit",
    seed=42,
):
    """Run simulation, training, and evaluation for all conditions."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    main_root = get_safe_path(sim_data_root)
    ensure_dir(main_root)
    ind_dir = get_safe_path("simdata_ind", base_dir=main_root)
    cor_dir = get_safe_path("simdata_cor", base_dir=main_root)
    summary_dir = get_safe_path("total_result", base_dir=main_root)
    ensure_dir(ind_dir)
    ensure_dir(cor_dir)
    ensure_dir(summary_dir)

    print("\nOutput directories:")
    print(f"  Root: {main_root}")
    print(f"  Independent: {ind_dir}")
    print(f"  Correlated: {cor_dir}")
    print(f"  Summary: {summary_dir}")

    for cond in conditions_list:
        npairs = cond["npairs"]
        corr_str = "ind" if cond.get("correlation_type") == "independent" else "cor"
        weight_sign_str = "pos" if cond["weight_sign"] == 1.0 else "mix"
        cond_dir = f"{cond['npersons']}_{npairs}_{corr_str}_{weight_sign_str}"
        cond_path = get_safe_path(cond_dir, base_dir=cor_dir if corr_str == "cor" else ind_dir)
        ensure_dir(cond_path)

        correlation_sampler: Optional[Callable[[dict, int], np.ndarray]] = cond.get("correlation_sampler")
        for r in range(1, repeat + 1):
            print(f"\n===== Condition {cond_dir} | Replication {r} =====")
            repeat_dir = get_safe_path(f"repeat_{r}", base_dir=cond_path)
            ensure_dir(repeat_dir)

            corr_matrix = correlation_sampler(cond, r) if callable(correlation_sampler) else cond.get("correlation_matrix", None)
            sim = Sim_data_TIRT(
                npersons=cond["npersons"],
                ntraits=cond["ntraits"],
                nblocks_per_trait=cond["nblocks_per_trait"],
                nitems_per_block=cond["nitems_per_block"],
                correlation_matrix=corr_matrix,
                w_range=cond["w_range"],
                b_range=cond["b_range"],
                weight_sign=cond["weight_sign"],
                comb_blocks=cond.get("comb_blocks", "random"),
            )
            sim.simulate()
            sim.save_csv(path=repeat_dir)

            Y = sim.responses
            true_traits = sim.theta
            item_trait_map = torch.tensor(sim.item_trait_map, dtype=torch.long)
            pair_definitions = torch.tensor(sim.pair_definitions, dtype=torch.long)
            weight_sign_tensor = torch.tensor(sim.weight_sign_array, dtype=torch.float32)

            model = AE_TIRT(
                input_dim=Y.shape[1],
                latent_dim=true_traits.shape[1],
                item_trait_map=item_trait_map,
                pair_definitions=pair_definitions,
                weight_sign=weight_sign_tensor,
                weight_constraint=weight_constraint,
                link_function=link_function,
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_start_time = time.time()
            resolved_batch_size = _resolve_batch_size(cond["nitems_per_block"], batch_size)
            train_model(
                model=model,
                train_data=Y,
                optimizer_name="adam",
                batch_size=resolved_batch_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                device=device,
                early_stopping_patience=early_stopping_patience,
                penalty_weight=Y.shape[1] * penalty_weight_factor,
            )
            train_time = time.time() - train_start_time

            eval_start_time = time.time()
            metrics = evaluate_model(model, Y, true_traits, device)
            item_metrics = evaluate_item_parameters(model, {"w": sim.w, "b": sim.b}, sim.pair_definitions, sim.item_trait_map)
            eval_time = time.time() - eval_start_time

            conditions_with_meta = {
                "condition_id": cond_dir,
                "npersons": cond["npersons"],
                "ntraits": cond["ntraits"],
                "npairs": Y.shape[1],
                "cor": "correlated" if cond.get("correlation_type", "independent") == "correlated" else "independent",
                "nitems_per_block": cond["nitems_per_block"],
                "weight_sign": cond["weight_sign"],
                "correlation_type": cond.get("correlation_type", "independent"),
            }
            save_results(metrics, train_time + eval_time, conditions_with_meta, output_dir=cond_path, repeat=r, summary_dir=summary_dir)
            save_item_parameter_results(item_metrics, conditions_with_meta, output_dir=cond_path, repeat=r, summary_dir=summary_dir)

    print("\n===== All conditions completed =====")
