Cloud Qwen API Reward
=====================

Run the reward judge on a Qwen-compatible cloud endpoint without downloading VLM
weights. RLinf keeps the policy rollout and the cloud monitor separate: the
monitor receives a short image history, returns ``positive``, ``negative``, or
``unclear``, and the existing reward worker maps that label to a scalar.

Where the API Is Called
-----------------------

The only outbound request is made by ``_call_api`` in
``rlinf/models/embodiment/reward/dashscope_vlm_reward_model.py``. The runtime path
is:

.. code-block:: text

   EnvWorker.get_reward_model_output
       -> RewardWorker.compute_image_rewards
       -> DashScopeHistoryVLMRewardModel.compute_reward
       -> DashScopeHistoryVLMRewardModel._call_api
       -> provider /v1/chat/completions endpoint
       -> qwentrend_reward_parser
       -> scalar reward
       -> EnvWorker final reward

The model is registered as ``dashscope_history_vlm`` in
``rlinf/models/embodiment/reward/__init__.py``. The HTTP client uses the standard
library, so this path does not add an SDK dependency. Set ``api_url`` and
``api_model`` to values supported by your provider account; the example uses the
DashScope OpenAI-compatible endpoint, but both fields are configurable.

Configure Credentials
---------------------

Export the key only in the process environment. Never put a key in YAML, a
checkpoint, or an experiment manifest:

.. code-block:: bash

   export DASHSCOPE_API_KEY='<your-key>'
   export QWEN_API_MODEL='<provider-supported-qwen-vl-model>'

What this does: the reward worker reads the two variables when it starts. The
model name remains an environment variable because available cloud model names
and access permissions depend on the provider account.

Use the Cloud Reward Config
---------------------------

Start with the single-node Flexiv top-level configuration. Its committed defaults
use ``is_dummy: true``, so it does not connect to the arms or cameras:

.. code-block:: bash

   export FLEXIV_PI05_MODEL_PATH=/path/to/pi05_bi_flexiv_checkpoint
   export FLEXIV_TASK_DESCRIPTION='Put the object into the box.'
   export DASHSCOPE_API_KEY='<your-key>'
   export QWEN_API_MODEL='<provider-supported-qwen-vl-model>'

   bash examples/embodiment/run_realworld.sh \
       realworld_bi_flexiv_ppo_openpi_pi05_cloud_qwen

What this does: loads the local OpenPI pi0.5 policy, creates zero-valued dummy
Flexiv observations, calls the cloud judge for valid five-frame windows, computes
GAE, and runs the PPO actor update. It validates the software path without moving
hardware. The configuration is
``examples/embodiment/config/realworld_bi_flexiv_ppo_openpi_pi05_cloud_qwen.yaml``.

Run a hardware shadow session only after the dummy run passes:

.. code-block:: bash

   export EMBODIED_PATH="$PWD/examples/embodiment"
   export PYTHONPATH="$PWD:${PYTHONPATH}"

   python examples/embodiment/train_embodied_agent.py \
       --config-path "$EMBODIED_PATH/config" \
       --config-name realworld_bi_flexiv_ppo_openpi_pi05_cloud_qwen \
       env.train.override_cfg.is_dummy=false \
       env.train.override_cfg.dry_run=true \
       reward.reward_weight=0.0

What this does: connects and resets the Flexiv setup, reads real cameras, and
intercepts policy actions before they reach the robot. The API output is logged
as ``env/reward_model_output`` but does not affect PPO because
``reward.reward_weight=0.0``.

.. warning::

   Online RL starts only when you explicitly set ``is_dummy=false``,
   ``dry_run=false``, and a non-zero ``reward.reward_weight``. Verify hardware
   limits, camera order, API latency, and checkpoint rollback first.

After every safety gate passes, start with one PPO update:

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-path "$EMBODIED_PATH/config" \
       --config-name realworld_bi_flexiv_ppo_openpi_pi05_cloud_qwen \
       env.train.override_cfg.is_dummy=false \
       env.train.override_cfg.dry_run=false \
       reward.reward_weight=1.0 \
       runner.max_steps=1

What this does: sends policy actions to the Flexiv arms and uses Qwen reward for
one actor update. Keep an operator at the hardware E-stop.

.. code-block:: yaml

   reward:
     use_reward_model: true
     reward_mode: history_buffer
     history_reward_assign: true
     reward_weight: 0.0       # shadow mode
     env_reward_weight: 1.0
     model:
       model_type: dashscope_history_vlm
       api_key_env: DASHSCOPE_API_KEY
       api_url: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
       api_model: ${oc.env:QWEN_API_MODEL}
       api_timeout: 30.0
       api_max_retries: 2
       api_max_tokens: 16
       api_temperature: 0.0
       extra_view_index: 0
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
           input_interval: 5
           history_keys: [main_images, extra_view_images]
           input_on_done: false

Set ``extra_view_index`` to the wrist or third-person camera that you validated
offline. A history window with fewer than five frames receives
``interval_reward`` and does not make an API request.

How the Final Reward Is Produced
--------------------------------

The current Flexiv environment returns ``0.0`` from
``BiFlexivDualEnv.step()``. ``RealWorldEnv`` passes that value through as the
environment reward. The cloud path independently maps Qwen labels as configured:

.. code-block:: text

   positive ->  1.0
   negative -> -0.2
   unclear  ->  0.0
   invalid  ->  0.0

``EnvWorker`` combines the two sources with:

.. code-block:: text

   final_reward = env_reward_weight * env_reward
                + reward_weight * qwen_reward

The Flexiv top-level config sets ``env_reward_weight: 0.0`` and
``reward_weight: 1.0``, so Qwen is the effective learning signal. With
``history_reward_assign: true``, one judgment is assigned to the current step and
the preceding steps in the valid five-frame window. Steps without a valid window
receive ``interval_reward: 0.0``.

Request Format and Response Parsing
-----------------------------------

``_call_api`` sends one JSON request with a user message containing the task
prompt and JPEG ``data:image/jpeg;base64,...`` blocks. It expects an
OpenAI-compatible response whose first choice contains ``message.content``. The
content may be plain text or text blocks. ``qwentrend_reward_parser`` then maps
the last recognized label to the configured scalar values.

.. note::

   This integration is deliberately endpoint-compatible rather than tied to one
   provider SDK. If your endpoint uses a different request or response schema,
   adapt ``_call_api`` and ``_extract_message_text`` while keeping
   ``compute_reward`` unchanged.

Latency, Cost, and Failure Behavior
-----------------------------------

Cloud inference is synchronous in the reward worker. One environment with a
five-frame dual-view window produces one request; a batch of environments produces
one request per environment. Increase ``input_interval`` and keep
``api_max_retries`` small enough to avoid a queue that outlives the control loop.

Start with ``reward_weight: 0.0`` and inspect request latency, parser invalid-rate,
and operator agreement. On an API error after retries, the worker raises a
``RuntimeError`` so the run stops instead of silently training on an unknown
reward. Add an explicit independent fallback before enabling physical online RL.

Next Step
---------

After shadow validation, follow :doc:`Safety and Acceptance <safety>`. Only then
set a non-zero ``reward_weight`` in a Flexiv-specific configuration.
