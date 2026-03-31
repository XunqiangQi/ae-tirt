"""Command-line entry points for reproducible workflows."""

from __future__ import annotations

from ae_tirt.experiments import batch_simulate_and_train, generate_tirt_conditions


def run_all_experiments():
    """CLI entrypoint with paper-fixed Study-1 settings."""
    conditions = generate_tirt_conditions(
        npersons_levels=[300, 500, 1000],
        npairs_levels=[15, 30],
        correlation_type_levels=["independent", "correlated"],
        weight_sign_levels=[1.0, 0.5],
    )
    batch_simulate_and_train(
        conditions_list=conditions,
        repeat=50,
        sim_data_root="Sim_data_result",
        num_epochs=500,
        learning_rate=0.001,
        early_stopping_patience=20,
        penalty_weight_factor=1,
        weight_constraint="standardized",
        link_function="probit",
        seed=42,
    )

