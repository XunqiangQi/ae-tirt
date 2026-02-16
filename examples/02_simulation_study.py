"""Example 2: simulation study with a reduced condition set."""

from ae_tirt.experiments import batch_simulate_and_train, generate_tirt_conditions


def main():
    conditions = generate_tirt_conditions(
        npersons_levels=[300],
        npairs_levels=[15],
        correlation_type_levels=["independent", "correlated"],
        weight_sign_levels=[1.0, 0.5],
    )
    batch_simulate_and_train(
        conditions_list=conditions,
        repeat=2,
        sim_data_root="example_outputs/study1_small",
        num_epochs=50,
        batch_size=16,
        learning_rate=1e-3,
        early_stopping_patience=10,
        penalty_weight_factor=0.1,
        weight_constraint="standardized",
        link_function="probit",
        seed=42,
    )


if __name__ == "__main__":
    main()
