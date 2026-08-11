Online RL Integration
=====================

Connect the Flexiv monitor through RLinf's reward worker. Use ``history_vlm`` for a
local Qwen3-VL checkpoint or :doc:`Cloud Qwen API <cloud_api>` for a cloud endpoint.
Keep the monitor in shadow mode until its labels, latency, and failure behavior are
measured on the target robot node.

Observation mapping
-------------------

``RealWorldEnv`` exposes the head camera as ``main_images``, the remaining cameras as
``extra_view_images``, and the task text as ``task_descriptions``. QwenTrend requires
two synchronized 5-frame videos. If the default extra-view stack contains two wrist
cameras, choose one deterministically in the input builder and document the choice.

Reward configuration
--------------------

Use the existing reward model keys as a starting point for local inference:

.. code-block:: yaml

   reward:
     use_reward_model: true
     group_name: "RewardGroup"
     reward_mode: history_buffer
     history_reward_assign: true
     reward_weight: 0.0
     env_reward_weight: 1.0
     model:
       model_path: /path/to/Qwen3-VL-4B-Instruct
       model_type: history_vlm
       lora_path: /path/to/flexiv-qwentrend-lora
       precision: bf16
       input_builder_name: qwentrend_input_builder
       input_builder_params:
         default_task_description: "<Flexiv task description>"
       reward_parser_name: qwentrend_reward_parser
       reward_parser_params:
         positive_reward: 1.0
         negative_reward: -0.2
         unclear_reward: 0.0
         invalid_reward: 0.0
       history_buffers:
         history_window:
           history_size: 5
           min_history_size: 5
           input_interval: 1
           history_keys:
             - main_images
             - extra_view_images
           input_on_done: false
       interval_reward: 0.0
       infer_micro_batch_size: 1
       max_new_tokens: 16
       do_sample: false
       temperature: 0.0
       use_chat_template: true

.. warning::

   This block is ready only after the Flexiv camera adapter presents exactly one
   extra view to ``qwentrend_input_builder``. With the current two-wrist observation,
   implement and select a Flexiv-specific input builder instead. Keep
   ``reward_weight: 0.0`` for the shadow run.

What the fields do:

.. list-table::
   :header-rows: 1
   :widths: 30 52 18

   * - Field
     - Effect
     - Flexiv starting point
   * - ``history_size`` / ``min_history_size``
     - Require a complete temporal window before inference.
     - ``5`` / ``5``
   * - ``input_interval``
     - Run the monitor every N environment steps.
     - Start at ``1``; increase after latency tests
   * - ``infer_micro_batch_size``
     - Bound GPU memory used by reward inference.
     - ``1`` for one real robot
   * - ``reward_weight``
     - Scale the learned reward in final reward computation.
     - Start at ``0`` in shadow mode
   * - ``env_reward_weight``
     - Preserve an independent environment reward if one exists.
     - Keep ``1`` if available; otherwise document why it is ``0``

The current EnvWorker combines external reward as
``env_reward_weight * env_reward + reward_weight * reward_model_output``. With
``history_reward_assign: true``, the history result is assigned to the corresponding
window according to the history manager. Read the implementation before changing
these semantics for a new Flexiv task.

Shadow mode
-----------

Do not let the monitor change actions, resets, or the reward used for updates during
the first online run. Add a logging-only path that records the raw generation,
parsed label, scalar mapping, timestamp, camera IDs, and operator decision. If you
reuse the RL runner, set ``reward_weight: 0.0`` and keep the reward worker enabled so
latency and outputs are still exercised.

Top-level Flexiv config
-----------------------

Use
``examples/embodiment/config/realworld_bi_flexiv_ppo_openpi_pi05_cloud_qwen.yaml``.
It includes:

.. list-table::
   :header-rows: 1
   :widths: 34 46 20

   * - Section
     - Required content
     - Source
   * - ``env.train`` / ``env.eval``
     - ``env_type: realworld`` and the Flexiv environment include.
     - ``env/realworld_bi_flexiv_tcp.yaml``
   * - ``cluster``
     - One robot node group and one or more GPU placements.
     - Heterogeneous cluster guide
   * - ``actor`` / ``rollout``
     - The policy checkpoint and action shape expected by Flexiv.
     - Policy-specific config
   * - ``reward``
     - The QwenTrend block above.
     - This page
   * - ``runner``
     - A short evaluation-only or shadow experiment first.
     - Existing embodied runners

The committed config runs with ``is_dummy: true``. Follow
:doc:`Cloud Qwen API <cloud_api>` for dummy, hardware-shadow, and online-RL launch
commands.

Next step
---------

Use :doc:`Safety and Acceptance <safety>` before setting a non-zero reward weight or
allowing policy updates on the physical robot.
