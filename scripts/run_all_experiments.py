"""Run full Study-1 AE-TIRT experiments with paper-fixed hyperparameters."""

from __future__ import annotations

import argparse

from ae_tirt.experiments import batch_simulate_and_train, generate_tirt_conditions


def main():
    parser = argparse.ArgumentParser(description="Run all TIRT simulation conditions (paper settings).")
    parser.add_argument(
        "--weight-constraint",
        type=str,
        default="standardized",
        choices=["standardized", "free"],
        help="Weight constraint used in estimation.",
    )
    parser.add_argument(
        "--link-function",
        type=str,
        default="probit",
        choices=["probit", "logit"],
        help="Link function used in decoder.",
    )
    args = parser.parse_args()

    # Fixed factorial design in the manuscript: 3 x 2 x 2 x 2.
    conditions = generate_tirt_conditions(
        npersons_levels=[300, 500, 1000],
        npairs_levels=[15, 30],
        correlation_type_levels=["independent", "correlated"],
        weight_sign_levels=[1.0, 0.5],
    )

    # Paper-fixed training and experiment settings.
    batch_simulate_and_train(
        conditions_list=conditions,
        repeat=50,
        sim_data_root="Sim_data_result",
        num_epochs=500,
        batch_size=16,
        learning_rate=0.001,
        early_stopping_patience=20,
        penalty_weight_factor=1,
        weight_constraint=args.weight_constraint,
        link_function=args.link_function,
        seed=42,
    )


if __name__ == "__main__":
    main()
