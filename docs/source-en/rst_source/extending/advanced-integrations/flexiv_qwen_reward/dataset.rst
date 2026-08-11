Data and Labels
===============

Build an episode-disjoint dataset whose labels describe progress, not merely the
last frame's success. This prevents the reward model from learning a shortcut such
as “gripper is closed” or “object is already at the goal.”

Raw episode contract
--------------------

Enable ``env.data_collection`` on the environment that records the Flexiv rollout.
The collector writes one ``.pkl`` episode per file. Preserve these fields:

.. list-table::
   :header-rows: 1
   :widths: 28 52 20

   * - Field
     - Meaning
     - Required
   * - ``observations``
     - Ordered observations containing ``main_images`` and the selected extra view.
     - Yes
   * - ``task`` or ``task_description``
     - Instruction used in the prompt.
     - Yes
   * - ``rewards`` / ``gae``
     - Progress signal for the existing automatic preprocessor.
     - Only for score-based labels
   * - ``episode_id``
     - Stable episode identifier for split and audit.
     - Yes
   * - timestamps and safety events
     - Camera freshness, operator intervention, stop, or fault information.
     - Strongly recommended

The Flexiv environment configuration already provides the task text and camera
settings in ``examples/embodiment/config/env/realworld_bi_flexiv_tcp.yaml``. Replace
all ``REPLACE_WITH_*`` values before collection and keep camera serial numbers out of
the public dataset manifest when they identify a physical installation.

Label provenance
----------------

Use this priority order:

.. list-table::
   :header-rows: 1
   :widths: 24 48 28

   * - Source
     - Use it for
     - Rule
   * - Human annotation
     - Initial zero-shot evaluation and ambiguous real-world cases.
     - Keep annotator, confidence, and notes.
   * - Task-state or instrumented signal
     - Large-scale training labels when the signal is independent of Qwen3-VL.
     - Version the rule and record its source.
   * - ``rewards`` / ``gae`` delta
     - Existing QwenTrend preprocessor.
     - Use only when the score is meaningful and not produced by the same VLM.
   * - Qwen3-VL pseudo-label
     - Triage or active-learning suggestions.
     - Never use as the only ground truth.

The current ``preprocess_qwentrend_reward_dataset.py`` computes each window's label
from the score delta, marks small deltas as ``unclear``, and can synthesize reversed
negative windows. That behavior is useful for simulation or instrumented tasks, but
it is not a substitute for human labels when Flexiv has no independent reward.

Window and split policy
-----------------------

Start with ``window_size: 5`` and synchronized frames. Keep the sampling interval
fixed during a comparison. Split by episode, then balance labels within each split.
Do not put a reversed augmentation and its source episode in different splits.

For score-based data, the existing command is:

.. code-block:: bash

   python examples/reward/preprocess_qwentrend_reward_dataset.py \
       --raw-data-path logs/<run>/collected_data \
       --output-dir logs/<run>/processed_qwentrend_reward_data \
       --window-size 5 \
       --stride 1

What this does: slices dual-view windows, writes per-sample pickle files and JSONL
manifests, and creates episode-disjoint ``train`` / ``eval`` splits. Do not run it
against a Flexiv dataset until ``rewards`` or ``gae`` has an independently justified
meaning.

Manual manifest adapter
-----------------------

Add a small adapter for human labels before LoRA training. Keep its output compatible
with ``qwentrend_progress_sft``. Each JSONL row should point to one pickle payload:

.. code-block:: json

   {
     "task": "Place the bottle in the carton.",
     "question": "<the versioned QwenTrend prompt>",
     "answer": "positive",
     "pkl_path": "train/pkl/positive_episode_0007_frames_0040_0044.pkl",
     "segment_metadata": {
       "episode_id": "episode_0007",
       "start_step": 40,
       "end_step": 44,
       "annotator": "operator_a",
       "confidence": 0.9
     }
   }

The referenced pickle payload must contain ``main_frames`` and
``extra_view_frames`` arrays. Keep ``answer`` exactly equal to the target label.
Store annotation metadata outside the prompt so you can revise the prompt without
rewriting the labels.

Dataset acceptance
------------------

Before training, check that every sample has five frames per view, valid RGB shapes,
one task string, one of the three labels, and an episode identifier. Report label
counts, missing-camera counts, duplicate sample IDs, and the number of safety events.
Reject a split with a camera failure or operator stop unless it is intentionally
included as an ``unclear`` or failure case.

Next step
---------

Point ``DUALVIEW_SFT_DATA_ROOT`` at the accepted dataset and follow
:doc:`LoRA Fine-Tuning <train>`.
