Methodology
===========

AE-TIRT combines amortized inference with a theory-constrained decoder.

Core decoder equation (single-trait-per-statement case):

.. math::

   P(y_{i,st}=1 \mid 	heta_{ia}, 	heta_{ib}) = \Phi\left(w_s	heta_{ia} - w_t	heta_{ib} + b_{st}ight)

where:

- :math:`w_s, w_t` are statement loadings,
- :math:`b_{st}` is a pair-specific intercept,
- :math:`\Phi(\cdot)` is the probit link.

Objective
---------

The training objective is:

.. math::

   \mathcal{L} = \mathcal{L}_{	ext{BCE}} + \lambda \cdot rac{1}{N}\sum_i \|	heta_i\|_2^2

This corresponds to reconstruction loss with latent-norm regularization,
compatible with MAP-style estimation under Gaussian latent priors.

Scope and trade-offs
--------------------

- Strength: fast estimation and scoring in large simulation grids.
- Limitation: no direct posterior intervals in current implementation.
- Recommended use: exploratory and operational FC scoring workflows.
