Simulation Protocol
===================

The default manuscript-aligned simulation design is a 3 x 2 x 2 x 2 factorial:

- sample size: ``[300, 500, 1000]``
- number of pairs: ``[15, 30]``
- trait covariance type: ``[independent, correlated]``
- loading sign pattern: ``[1.0, 0.5]`` (all positive vs mixed)

Replications
------------

- 50 replications per condition
- total runs: 24 x 50 = 1200

Fixed training settings
-----------------------

- ``num_epochs = 500``
- ``batch_size = 16``
- ``learning_rate = 0.001``
- ``early_stopping_patience = 20``
- ``penalty_weight_factor = 1``
- ``seed = 42``

Launch
------

.. code-block:: bash

   python scripts/run_all_experiments.py
