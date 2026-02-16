Quickstart
==========

This quickstart provides two standard workflows:

1) synthetic-data simulation and analysis,
2) externally provided data analysis.

Workflow 1: simulate then analyze
---------------------------------

This pipeline reproduces the AE-TIRT synthetic-data process in four steps:

1. simulate forced-choice data under TIRT,
2. instantiate AE-TIRT model,
3. fit model via mini-batch optimization,
4. evaluate trait recovery.

.. code-block:: python

   import torch
   from ae_tirt import AE_TIRT, Sim_data_TIRT, train_model, evaluate_model

   sim = Sim_data_TIRT(npersons=300, ntraits=5, nblocks_per_trait=6, nitems_per_block=2).simulate()

   model = AE_TIRT(
       input_dim=sim.responses.shape[1],
       latent_dim=sim.theta.shape[1],
       item_trait_map=torch.tensor(sim.item_trait_map, dtype=torch.long),
       pair_definitions=torch.tensor(sim.pair_definitions, dtype=torch.long),
       weight_sign=torch.tensor(sim.weight_sign_array, dtype=torch.float32),
   )

   history = train_model(model=model, train_data=sim.responses, num_epochs=100)
   metrics = evaluate_model(model, sim.responses, sim.theta)
   print(metrics["traits"]["overall"])

For paper-aligned settings, use:

.. code-block:: bash

   python scripts/run_all_experiments.py

Workflow 2: externally provided data
------------------------------------

Use this when input files are provided by external collaborators:

.. code-block:: bash

   python scripts/run_real_data_analysis.py --data-dir data/real

The script performs schema and consistency checks before fitting. See
``empirical_protocol`` for exact file requirements.
