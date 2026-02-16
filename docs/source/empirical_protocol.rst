Empirical Protocol
==================

This workflow supports empirical FC datasets where each row is a respondent
and each column is a binary paired-comparison response.

Required inputs
---------------

- ``responses.csv``: response matrix of shape ``N x L`` with values in ``{0,1}``.
- ``pair_definitions.csv``: exactly ``L`` rows and columns ``item1,item2``.
- ``item_trait_map.csv``: exactly ``D`` rows and column ``item_trait``.
- ``weight_sign.csv``: exactly ``D`` rows and column ``weight_sign`` (usually ``+1/-1``).

Data-format constraints
-----------------------

Before fitting, the script validates:

- response values are strictly binary (0/1),
- number of response columns equals number of rows in ``pair_definitions.csv``,
- all pair indices are valid 1-based item IDs in ``1..D``,
- ``weight_sign.csv`` length equals ``item_trait_map.csv`` length.

If any rule is violated, the script exits with an explicit error message.
Template files are provided in ``data/sample/`` for collaborator handoff.

Script
------

.. code-block:: bash

   python scripts/run_real_data_analysis.py

Paper dataset in this workspace
-------------------------------

.. code-block:: bash

   python scripts/run_real_data_analysis.py \
     --data-dir data/real \
     --responses-file X_responses.csv \
     --batch-size 16 \
     --num-epochs 500 \
     --learning-rate 0.001 \
     --early-stopping-patience 20 \
     --penalty-weight-factor 1.0

or:

.. code-block:: bash

   python examples/05_paper_real_data_example.py

Optional arguments:

.. code-block:: bash

   python scripts/run_real_data_analysis.py \
     --data-dir data/real \
     --batch-size 16 \
     --num-epochs 500 \
     --learning-rate 0.001 \
     --early-stopping-patience 20 \
     --penalty-weight-factor 1.0 \
     --optimizer adam \
     --seed 42 \
     --weight-constraint standardized \
     --link-function probit

Long-format input option
------------------------

If responses are provided as R-compatible long table (``Real_data.csv`` with
``person,itemC,response``), enable:

.. code-block:: bash

   python scripts/run_real_data_analysis.py --data-dir data/real --use-long-format

Bundled empirical files and source notes are documented in ``data/real/README.md``.

Recommended reporting
---------------------

- trait-level agreement with benchmark estimators,
- item-parameter agreement (loading/intercept correlations),
- runtime and numerical stability indicators,
- reproducibility metadata (seed, software versions, hardware).
