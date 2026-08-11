Flexiv QwenTrend Reward Workflow
================================

Use this collection to turn a Flexiv camera stream into a monitored, conservative
learning signal. Start with offline zero-shot judging, then fine-tune Qwen3-VL only
if the held-out data shows that the base model is not reliable enough.

.. important::

   This is an implementation blueprint for the Flexiv integration. RLinf already
   contains the ``bi_flexiv`` environment, local and cloud history reward workers,
   a Flexiv OpenPI pi0.5 top-level config, QwenTrend input/parser utilities, and the
   QwenTrend SFT dataset. It does **not** yet contain a standalone zero-shot
   evaluation CLI or a manual-label adapter. Add those pieces before relying on the
   judge for physical online RL.

Architecture
------------

Keep the policy and the monitor separate:

.. code-block:: text

   Flexiv cameras + task text
            |
            v
   head view + selected wrist view, 5 frames
            |
            v
   Qwen3-VL (+ optional reward LoRA)
            |
            v
   positive / negative / unclear
            |
            v
   scalar reward -> RLinf EnvWorker -> policy update

Qwen3-VL is a general VLM, not a vendor-provided robot reward model. The
Flexiv-specific component is the prompt, camera adapter, label policy, and optional
LoRA checkpoint that you train from Flexiv data.

Phased rollout
--------------

.. list-table::
   :header-rows: 1
   :widths: 14 28 38 20

   * - Phase
     - Goal
     - Required evidence
     - Policy impact
   * - 0
     - Hardware and camera dry run
     - Stable ``main_images``, selected wrist view, task text, and timestamps.
     - None
   * - 1
     - Zero-shot judge
     - Human-labeled 5-frame windows, confusion matrix, invalid-output rate, and latency.
     - Shadow only
   * - 2
     - Reward dataset and LoRA
     - Episode-disjoint train/eval split and held-out comparison against zero-shot.
     - Shadow only
   * - 3
     - Online reward shadow mode
     - Reward logs agree with the operator without changing resets or actions.
     - None
   * - 4
     - Conservative online RL
     - Safety checklist passed and rollback checkpoint available.
     - Limited updates

Choose a phase:

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: Zero-Shot Validation
      :link: zero_shot
      :link-type: doc

      Test the base Qwen3-VL judge before training anything.

   .. grid-item-card:: Data and Labels
      :link: dataset
      :link-type: doc

      Define the Flexiv episode contract and prevent label leakage.

   .. grid-item-card:: LoRA Fine-Tuning
      :link: train
      :link-type: doc

      Fine-tune Qwen3-VL only when the zero-shot gate fails.

   .. grid-item-card:: Online RL Integration
      :link: online_rl
      :link-type: doc

      Connect ``history_vlm`` to the reward channel in shadow mode first.

   .. grid-item-card:: Cloud Qwen API
      :link: cloud_api
      :link-type: doc

      Call a configurable Qwen-compatible endpoint from the reward worker.

   .. grid-item-card:: Safety and Acceptance
      :link: safety
      :link-type: doc

      Gate physical experiments with explicit stop and rollback criteria.

Current RLinf components
------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 36 24

   * - Component
     - Path
     - Status
   * - Flexiv environment
     - ``examples/embodiment/config/env/realworld_bi_flexiv_tcp.yaml``
     - Available
   * - Observation wrapper
     - ``rlinf/envs/realworld/realworld_env.py``
     - Available
   * - QwenTrend preprocessor
     - ``examples/reward/preprocess_qwentrend_reward_dataset.py``
     - Available; expects score-based labels
   * - QwenTrend SFT config
     - ``examples/sft/config/qwen3vl_sft_qwentrend.yaml``
     - Available
   * - Online reward model
     - ``model_type: history_vlm``
     - Available for local VLM weights; ``dashscope_history_vlm`` is available for a cloud endpoint
   * - Flexiv top-level RL config
     - ``realworld_bi_flexiv_ppo_openpi_pi05_cloud_qwen.yaml``
     - Available; defaults to dummy mode

.. toctree::
   :hidden:

   Zero-Shot Validation <zero_shot>
   Data and Labels <dataset>
   LoRA Fine-Tuning <train>
   Online RL Integration <online_rl>
   Cloud Qwen API <cloud_api>
   Safety and Acceptance <safety>
