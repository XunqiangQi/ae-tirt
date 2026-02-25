"""Example 1: basic end-to-end AE-TIRT workflow on synthetic data."""

import torch

from ae_tirt import AE_TIRT, Sim_data_TIRT, evaluate_model, train_model

# ---------------------------------------------------------------------------
# Standard-error computation flag
#
# SE estimation is deliberately disabled by default.  In academic practice,
# SEs derived from the observed information matrix are a *post-estimation*
# inferential step distinct from model fitting.  Computing the full Hessian
# (use_hessian_diag=False) scales as O(P²) in the total number of free
# parameters and can be very slow for large samples; the diagonal
# approximation (use_hessian_diag=True) is faster but yields uncorrelated
# parameter uncertainty only.  Enable this flag only when you require SE /
# CI / Wald-z values for publication reporting.
# ---------------------------------------------------------------------------
COMPUTE_SE = True


def main():
    sim = Sim_data_TIRT(
        npersons=500,
        ntraits=5,
        nblocks_per_trait=12,
        nitems_per_block=3,
        weight_sign=0.5,
        w_range=(0.65, 0.95),
        b_range=(-1.0, 1.0),
        comb_blocks="random",
    ).simulate()

    model = AE_TIRT(
        input_dim=sim.responses.shape[1],
        latent_dim=sim.theta.shape[1],
        item_trait_map=torch.tensor(sim.item_trait_map, dtype=torch.long),
        pair_definitions=torch.tensor(sim.pair_definitions, dtype=torch.long),
        weight_sign=torch.tensor(sim.weight_sign_array, dtype=torch.float32),
        weight_constraint="standardized",
        link_function="probit",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history = train_model(
        model=model,
        train_data=sim.responses,
        optimizer_name="adam",
        batch_size=32,
        num_epochs=500,
        learning_rate=1e-3,
        device=device,
        early_stopping_patience=20,
        penalty_weight=sim.responses.shape[1] * 1.0,
    )
    metrics = evaluate_model(model, sim.responses, sim.theta, device=device)

    print(f"Trained epochs: {len(history['total_loss'])}")
    print(f"Overall trait RMSE: {metrics['traits']['overall']['rmse']:.4f}")
    print(f"Overall trait correlation: {metrics['traits']['overall']['cor']:.4f}")

    # -----------------------------------------------------------------------
    # Optional: post-estimation standard errors
    #
    # Methodology (Mislevy, 1986; Baker & Kim, 2004):
    #   SEs are computed from the observed Fisher information I(θ̂) = -∂²ℓ/∂θ²
    #   evaluated at the MAP/point estimates obtained after training.
    #   Var(θ̂) ≈ I(θ̂)⁻¹  →  SE(θ̂_k) = sqrt[ I(θ̂)⁻¹_kk ]
    #
    # Parameters
    # ----------
    # method          : "observed" uses the Hessian of the log-likelihood;
    #                   "expected" is equivalent here (both are post-hoc).
    # use_hessian_diag: True  → diagonal approximation (fast, O(P));
    #                   False → full Hessian inversion (exact, O(P²) memory).
    # regularization  : small ridge added to I(θ̂) for numerical stability
    #                   (default 1e-4 is suitable for most configurations).
    #
    # The resulting report follows the publication convention:
    #   estimate | SE | z (Wald) | p (two-sided) | 95% CI lower | 95% CI upper
    # -----------------------------------------------------------------------
    if COMPUTE_SE:
        responses_device = torch.tensor(sim.responses, dtype=torch.float32).to(device)
        se_results = model.compute_standard_errors(
            x=responses_device,
            method="observed",
            use_hessian_diag=True,   # set False for full Hessian (slow)
            regularization=1e-4,
            verbose=True,
        )
        model.print_standard_errors_summary(se_results, num_examples=5, alpha=0.05)


if __name__ == "__main__":
    main()
