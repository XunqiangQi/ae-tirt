"""Example 1: basic end-to-end AE-TIRT workflow on synthetic data."""

import torch

from ae_tirt import AE_TIRT, Sim_data_TIRT, evaluate_model, train_model

# ---------------------------------------------------------------------------
# Standard-error computation flag  (default: False)
#
# SE estimation is an optional, post hoc step and is disabled by default.
# SEs are derived from the observed information matrix (negative Hessian) of
# the decoder log-likelihood, yielding asymptotic, approximate SEs
# conditional on the encoder-based trait estimates.  Two modes are supported:
#   - Full Hessian  (use_hessian_diag=False): more accurate, O(P²) cost.
#   - Diagonal approximation (use_hessian_diag=True): faster, approximate.
# For standardized loadings (w = sigmoid(u) × sign), SEs on the loading scale
# are obtained via the chain rule (delta method).
# Enable COMPUTE_SE only when SE / CI / Wald-z values are needed for reporting.
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
    # Optional: post hoc standard errors via IRT Fisher information
    #
    # SEs are computed from the observed Fisher information evaluated at the
    # item response likelihood — the Bernoulli log-likelihood of observed
    # forced-choice responses under the decoder:
    #
    #   I_obs(θ̂, â, b̂) = -∂² log L(X | θ̂, â, b̂) / ∂ψ²
    #
    # where ψ = (θ, a, b) denotes person traits, statement loadings, and
    # pair intercepts jointly.  Asymptotic SEs follow from the inverse
    # observed information:
    #
    #   SE(ψ̂_k) = sqrt[ I_obs(ψ̂)⁻¹_kk ]
    #
    # These SEs are conditional on the encoder-derived trait estimates (θ̂)
    # and do *not* account for uncertainty propagated through the encoder;
    # they are analogous to post hoc SEs in standard IRT software.
    # For standardized loadings (â = sigmoid(u) × sign), SEs are mapped to
    # the loading scale via the delta method (chain rule on sigmoid).
    #
    # Key arguments
    # -------------
    # method          : "observed" — uses the item response log-likelihood
    #                   Hessian; appropriate for IRT parameter inference.
    # use_hessian_diag: False → full I_obs inversion (accurate, O(P²) cost);
    #                   True  → diagonal approximation (fast, ignores
    #                           off-diagonal parameter covariances).
    # regularization  : ridge term added to I_obs for numerical stability
    #                   (default 1e-4; increase if inversion is ill-conditioned).
    #
    # Report columns (publication convention):
    #   estimate | SE | z (Wald) | p (two-sided) | CI lower | CI upper
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
