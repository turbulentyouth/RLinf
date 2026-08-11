Zero-Shot Validation
====================

Measure whether the base Qwen3-VL can judge Flexiv progress before you create a
reward dataset or change the policy. Run this stage offline and in shadow mode.

Input contract
--------------

Build each sample from one task instruction and two synchronized videos:

.. list-table::
   :header-rows: 1
   :widths: 28 52 20

   * - Field
     - Required content
     - RLinf source
   * - ``task``
     - One imperative sentence, for example ``Place the bottle in the carton.``
     - ``override_cfg.task_description``
   * - ``main_images``
     - Five RGB frames from the fixed head camera.
     - ``RealWorldEnv`` observation
   * - ``extra_view_images``
     - Five RGB frames from one fixed wrist or third-person camera.
     - Flexiv camera adapter
   * - ``label``
     - Human reference: ``positive``, ``negative``, or ``unclear``.
     - Annotation manifest

The current ``realworld_bi_flexiv_tcp.yaml`` exposes a head camera and two wrist
cameras through ``extra_view_images``. The online QwenTrend builder consumes two
videos, not an arbitrary stack of wrist views. Select one wrist view in a Flexiv
adapter, or add a dedicated ``flexiv_qwentrend_input_builder``. Do not silently
change camera order between collection, evaluation, and RL.

Prompt and output
-----------------

Use the same prompt in offline evaluation and online inference:

.. code-block:: text

   You are currently performing the task: <task>.
   You are given two synchronized 5-frame videos from different camera views
   (main view and third-person view) of the same robot action window. Judge whether
   the action trend is positive, negative, or unclear. Answer with exactly one word:
   positive, negative, or unclear.

The existing ``qwentrend_reward_parser`` accepts these labels, and also extracts
them from a JSON object with a ``trend`` field. Keep the exact labels in the
annotation data so parser errors remain visible.

Evaluation set
--------------

Create at least a few hundred windows across the target task's normal progress,
recoverable mistakes, stalls, and camera occlusions. Have an operator label every
window, then double-label a representative subset. Split by episode, never by
window, so adjacent frames cannot leak across train and evaluation sets.

Record the following measurements:

.. list-table::
   :header-rows: 1
   :widths: 34 48 18

   * - Measurement
     - Why it matters
     - Suggested gate
   * - Macro F1 over three labels
     - Prevents a majority-class ``unclear`` judge from looking good.
     - ``>= 0.85``
   * - Positive recall
     - Detects progress that should not be discarded.
     - ``>= 0.90``
   * - Invalid-output rate
     - Reveals prompt, tokenizer, or parser failures.
     - ``<= 1%``
   * - p95 inference latency
     - Determines whether ``input_interval`` is safe for the control loop.
     - Measure on target hardware

These are starting gates for a pilot, not RLinf guarantees. Tighten them after
you collect more tasks and operators.

Implementation gap
------------------

Add a small offline evaluator before using the workflow as a command-line recipe.
The proposed interface is:

.. code-block:: text

   examples/reward/eval_flexiv_qwentrend_zero_shot.py
       --manifest <jsonl>
       --model-path <Qwen3-VL directory>
       --output <metrics.json>

The evaluator should save raw generations, parsed labels, latency, camera metadata,
and a confusion matrix. Keep this tool separate from the RL runner so a bad judge
cannot move the robot.

Next step
---------

If the base model passes the held-out gate, keep ``lora_path`` unset and continue to
:doc:`Online RL Integration <online_rl>` in shadow mode. Otherwise, continue with
:doc:`Data and Labels <dataset>` and :doc:`LoRA Fine-Tuning <train>`.
