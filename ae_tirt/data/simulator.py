"""TIRT simulation data generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm

from ae_tirt.utils.paths import ensure_dir, get_safe_path


class Sim_data_TIRT:
    def __init__(
        self,
        npersons=100,
        ntraits=3,
        nblocks_per_trait=12,
        nitems_per_block=2,
        sigma=None,
        correlation_matrix=None,
        w_range=(0.5, 0.95),
        b_range=(-1, 1),
        weight_sign=None,
        comb_blocks="random",
    ):
        self.npersons = npersons
        self.ntraits = ntraits
        self.nblocks_per_trait = nblocks_per_trait
        self.nitems_per_block = nitems_per_block
        self.sigma = sigma if sigma is not None else np.eye(ntraits)
        self.w_range = w_range
        self.b_range = b_range
        self.comb_blocks = comb_blocks.lower()
        self.correlation_matrix = correlation_matrix
        self.weight_sign = weight_sign

        self.sim_data = None
        self.responses = None
        self.item_trait_map = None
        self.pair_definitions = None
        self.theta = None
        self.w = None
        self.b = None
        self.weight_sign_array = None

    def _make_trait_combs(self):
        assert (self.ntraits * self.nblocks_per_trait) % self.nitems_per_block == 0
        if self.comb_blocks == "fixed":
            traits = np.repeat(np.arange(self.ntraits), self.nblocks_per_trait)
            return traits.reshape(-1, self.nitems_per_block)
        nblocks = (self.ntraits * self.nblocks_per_trait) // self.nitems_per_block
        grid = [np.arange(self.ntraits) for _ in range(self.nitems_per_block)]
        all_combs = np.array(np.meshgrid(*grid)).T.reshape(-1, self.nitems_per_block)
        valid = all_combs[
            np.apply_along_axis(lambda r: len(np.unique(r)) == self.nitems_per_block, 1, all_combs)
        ]
        count = np.zeros(self.ntraits, dtype=int)
        chosen = []
        max_tries = 20 * nblocks
        for b in range(nblocks):
            for _ in range(max_tries):
                pick = valid[np.random.randint(len(valid))]
                tmp = count.copy()
                tmp[pick] += 1
                if tmp.max() <= tmp.min() + 1 and np.all(tmp[pick] <= self.nblocks_per_trait):
                    count = tmp
                    chosen.append(pick)
                    break
            else:
                raise RuntimeError(f"Cannot find valid combination for block {b}")
        return np.vstack(chosen)

    def simulate(self):
        trait_combs = self._make_trait_combs()
        nblocks = trait_combs.shape[0]
        pairs = []
        for blk, traits in enumerate(trait_combs, start=1):
            for i in range(self.nitems_per_block):
                for j in range(i + 1, self.nitems_per_block):
                    item1 = (blk - 1) * self.nitems_per_block + i + 1
                    item2 = (blk - 1) * self.nitems_per_block + j + 1
                    pairs.append(
                        {
                            "block": blk,
                            "item1": item1,
                            "item2": item2,
                            "trait1": traits[i] + 1,
                            "trait2": traits[j] + 1,
                        }
                    )
        pairs_df = pd.DataFrame(pairs)
        npairs = len(pairs_df)
        pairs_df["comparison"] = pairs_df.groupby("block").cumcount() + 1

        if self.correlation_matrix is not None:
            correlation_matrix = np.array(self.correlation_matrix)
            if correlation_matrix.shape != (self.ntraits, self.ntraits):
                raise ValueError(f"correlation_matrix must be {self.ntraits}x{self.ntraits}")
            self.sigma = correlation_matrix
        elif self.sigma is None:
            self.sigma = np.eye(self.ntraits)

        self.theta = multivariate_normal.rvs(mean=np.zeros(self.ntraits), cov=self.sigma, size=self.npersons)
        self.theta = np.clip(self.theta, -3, 3)

        D = nblocks * self.nitems_per_block
        pos_ratio = 1.0 if self.weight_sign is None else float(self.weight_sign)
        if not (0 <= pos_ratio <= 1):
            raise ValueError("weight_sign must be between 0 and 1")

        n_positive = int(round(D * pos_ratio))
        n_negative = D - n_positive
        self.weight_sign_array = np.concatenate([np.ones(n_positive, dtype=int), -np.ones(n_negative, dtype=int)])
        np.random.shuffle(self.weight_sign_array)

        w_magnitude = np.random.uniform(*self.w_range, size=D)
        self.w = w_magnitude * self.weight_sign_array
        self.b = np.random.uniform(*self.b_range, size=npairs)

        def tirt_prob(a, b, wj, wk, bias):
            return norm.cdf(wj * a - wk * b + bias)

        probs = np.zeros((self.npersons, npairs))
        for idx in range(npairs):
            row = pairs_df.iloc[idx]
            a = self.theta[:, row["trait1"] - 1]
            b_val = self.theta[:, row["trait2"] - 1]
            probs[:, idx] = tirt_prob(
                a,
                b_val,
                self.w[row["item1"] - 1],
                self.w[row["item2"] - 1],
                self.b[idx],
            )

        self.responses = np.random.binomial(1, probs)
        pairs_df = pairs_df.copy()
        pairs_df["itemC"] = np.arange(1, npairs + 1)
        pairs_df["sign1"] = self.weight_sign_array[pairs_df["item1"].values - 1]
        pairs_df["sign2"] = self.weight_sign_array[pairs_df["item2"].values - 1]

        pairs_expanded = pd.concat([pairs_df] * self.npersons, ignore_index=True)
        self.sim_data = pd.DataFrame(
            {
                "person": np.repeat(np.arange(1, self.npersons + 1), npairs),
                "block": pairs_expanded["block"].values,
                "comparison": pairs_expanded["comparison"].values,
                "itemC": pairs_expanded["itemC"].values,
                "trait1": pairs_expanded["trait1"].values,
                "trait2": pairs_expanded["trait2"].values,
                "item1": pairs_expanded["item1"].values,
                "item2": pairs_expanded["item2"].values,
                "sign1": pairs_expanded["sign1"].values,
                "sign2": pairs_expanded["sign2"].values,
                "response": self.responses.flatten(),
            }
        )

        self.item_trait_map = trait_combs.flatten(order="C") + 1
        self.pair_definitions = pairs_df[["item1", "item2"]].values
        return self

    def save_csv(self, path=None):
        base = get_safe_path("simdata") if path is None else path
        ensure_dir(base)

        self.sim_data.to_csv(get_safe_path("sim_data.csv", base_dir=base), index=False)
        # Compatibility alias: long-format file aligned with common R thurstonian workflows.
        self.sim_data.to_csv(get_safe_path("Real_data.csv", base_dir=base), index=False)
        pd.DataFrame(self.responses).to_csv(get_safe_path("responses.csv", base_dir=base), index=False)
        # Compatibility alias: matrix response file with explicit user id column.
        x_df = pd.DataFrame(self.responses)
        x_df.insert(0, "userid", np.arange(1, self.npersons + 1))
        x_df.to_csv(get_safe_path("X_responses.csv", base_dir=base), index=False)
        pd.DataFrame({"item_trait": self.item_trait_map}).to_csv(
            get_safe_path("item_trait_map.csv", base_dir=base), index=False
        )
        pd.DataFrame({"statement_id": np.arange(1, len(self.item_trait_map) + 1), "trait_id": self.item_trait_map}).to_csv(
            get_safe_path("item_trait_map_thurstonian.csv", base_dir=base), index=False
        )
        pd.DataFrame(self.pair_definitions, columns=["item1", "item2"]).to_csv(
            get_safe_path("pair_definitions.csv", base_dir=base), index=False
        )
        pd.DataFrame(self.pair_definitions, columns=["statement_j", "statement_k"]).to_csv(
            get_safe_path("pair_definitions_thurstonian.csv", base_dir=base), index=False
        )

        pd.DataFrame(self.theta, columns=[f"theta_{i+1}" for i in range(self.ntraits)]).to_csv(
            get_safe_path("true_theta.csv", base_dir=base), index=False
        )
        pd.DataFrame({"w": self.w}).to_csv(get_safe_path("true_w.csv", base_dir=base), index=False)
        pd.DataFrame({"b": self.b}).to_csv(get_safe_path("true_b.csv", base_dir=base), index=False)
        pd.DataFrame({"weight_sign": self.weight_sign_array}).to_csv(
            get_safe_path("weight_sign.csv", base_dir=base), index=False
        )
        pd.DataFrame({"statement_id": np.arange(1, len(self.weight_sign_array) + 1), "weight_sign": self.weight_sign_array}).to_csv(
            get_safe_path("weight_sign_thurstonian.csv", base_dir=base), index=False
        )
