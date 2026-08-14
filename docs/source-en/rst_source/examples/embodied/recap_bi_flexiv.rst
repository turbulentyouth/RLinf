Offline RECAP for Bi-Flexiv Records
====================================

Train a new π₀.₅ checkpoint from Bi-Flexiv LeRobot records without connecting
RLinf to the robot. RECAP first builds trajectory returns, trains its local
SigLIP2 + Gemma3 value model, computes per-frame advantages, and then optimizes
the OpenPI policy with classifier-free guidance.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Algorithm
      :text-align: center

      RECAP + CFG

   .. grid-item-card:: Models
      :text-align: center

      RECAP value VLM · π₀.₅

   .. grid-item-card:: Environments / Data
      :text-align: center

      Bi-Flexiv · LeRobot v3

   .. grid-item-card:: Training
      :text-align: center

      Offline · four stages

| **You'll do:** validate the record → compute returns → train the value VLM → compute advantages → train π₀.₅.
| **Prerequisites:** OpenPI environment · SigLIP2 checkpoint · Gemma3 checkpoint · π₀.₅ PyTorch checkpoint.

LeRobot Version
---------------

The Flexiv recorder emits the LeRobot v3 file-based format. If the OpenPI
environment contains ``lerobot==0.3.3``, replace it once with the pinned v3
reader before training:

.. code:: bash

   uv pip install --force-reinstall --no-deps \
     "lerobot @ git+https://github.com/huggingface/lerobot.git@v0.4.4"

Verify the active environment:

.. code:: bash

   python -c "from importlib.metadata import version; print(version('lerobot'))"

The output must be ``0.4.4`` for LeRobot v3 records. RLinf now opens the local
dataset with an explicit ``root`` and reports a direct version error instead of
falling back to a nonexistent Hugging Face Hub repository.

Prepare the Dataset
-------------------

The Flexiv recorder output is already LeRobot v3. Extract and validate it:

.. code:: bash

   python examples/offline_rl/prepare_flexiv_recap.py \
     --input-zip /path/to/stack-cubes-flexiv.zip \
     --output-dir /data/flexiv-recap

Use the dataset path printed by the command. It points to the directory that
directly contains ``data/``, ``meta/``, and ``videos/``.

.. warning::

   The supplied record contains one episode. The commands below treat it as a
   successful expert demonstration (``type=sft``). This is enough to validate
   the pipeline, but useful policy improvement normally requires multiple
   successful and failed episodes covering different states and actions.

Run It
------

Set the paths once:

.. code:: bash

   export FLEXIV_DATA=/data/flexiv-recap/stack-cubes-flexiv-0811
   export SIGLIP_PATH=/models/siglip2-so400m-patch14-224
   export GEMMA_PATH=/models/gemma-3-270m
   export PI05_PATH=/models/pi05_base_pytorch

Step 1 writes ``meta/returns_success.parquet``:

.. code:: bash

   bash examples/offline_rl/advantage_labeling/recap/process/run_compute_returns.sh \
     recap_bi_flexiv_compute_returns \
     data.train_data_paths.0.dataset_path=$FLEXIV_DATA

Step 2 trains the local RECAP value VLM:

.. code:: bash

   bash examples/offline_rl/advantage_labeling/recap/run_value_sft.sh \
     recap_bi_flexiv_value_model_sft \
     data.train_data_paths.0.dataset_path=$FLEXIV_DATA \
     actor.model.siglip_path=$SIGLIP_PATH \
     actor.model.gemma3_path=$GEMMA_PATH \
     actor.model.tokenizer_path=$GEMMA_PATH

Set ``VALUE_CKPT`` to the resulting ``global_step_N/actor`` directory. Step 3
uses the VLM to write ``meta/advantages_recap.parquet``:

.. code:: bash

   export VALUE_CKPT=/path/to/value_sft/checkpoints/global_step_N/actor

   bash examples/offline_rl/advantage_labeling/recap/process/run_compute_advantages.sh \
     recap_bi_flexiv_compute_advantages --nproc 1 \
     data.train_data_paths.0.dataset_path=$FLEXIV_DATA \
     advantage.value_checkpoint=$VALUE_CKPT \
     advantage.model.siglip_path=$SIGLIP_PATH \
     advantage.model.gemma3_path=$GEMMA_PATH \
     advantage.model.tokenizer_path=$GEMMA_PATH

Step 4 optimizes π₀.₅ from the advantage labels:

.. code:: bash

   bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh \
     cfg_rl_bi_flexiv \
     data.train_data_paths.0.dataset_path=$FLEXIV_DATA \
     actor.model.model_path=$PI05_PATH

The updated policy is stored under
``logs/cfg_rl/cfg_rl_bi_flexiv-*/checkpoints/global_step_N/actor``. Load that
actor checkpoint in the Flexiv OpenPI inference process; RLinf is not required
at deployment time.

How the Reward Works
--------------------

RECAP's local VLM is a learned value function, not a zero-shot reward judge.
Step 1 initializes rewards from trajectory outcome and time-to-completion. Step
2 trains the VLM to predict the corresponding return from camera images and the
task prompt. Step 3 combines its predictions with N-step rewards to produce
advantages. Step 4 uses positive and negative advantage labels to update π₀.₅.
