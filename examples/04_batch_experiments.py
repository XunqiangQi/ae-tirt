"""Example 4: multi-study batch experiment launcher."""

from ae_tirt.experiments import (
    batch_simulate_and_train,
    generate_tirt_conditions,
    generate_tirt_conditions_study2,
)


def main():
    study1_conditions = generate_tirt_conditions(
        npersons_levels=[300, 500],
        npairs_levels=[15, 30],
        correlation_type_levels=["independent", "correlated"],
        weight_sign_levels=[1.0, 0.5],
    )
    study2_conditions = generate_tirt_conditions_study2(
        ntraits_levels=[10],
        npersons=500,
        nblocks_per_trait=9,
        nitems_per_block=2,
        weight_sign=0.5,
        correlation_range=(-0.5, 0.5),
    )

    batch_simulate_and_train(
        conditions_list=study1_conditions + study2_conditions,
        repeat=1,
        sim_data_root="example_outputs/all_studies",
        num_epochs=500,
        learning_rate=1e-3,
        early_stopping_patience=20,
        penalty_weight_factor=0.1,
        weight_constraint="standardized",
        link_function="probit",
        seed=42,
    )


if __name__ == "__main__":
    main()
