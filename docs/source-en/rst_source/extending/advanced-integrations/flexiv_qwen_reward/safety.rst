Safety and Acceptance
=====================

Treat the VLM as a progress monitor, never as an emergency-stop controller. A
Qwen3-VL output can be stale, malformed, or visually plausible but wrong. Keep
hardware limits and operator intervention outside the model.

Required safeguards
-------------------

.. list-table::
   :header-rows: 1
   :widths: 34 46 20

   * - Safeguard
     - Requirement
     - Evidence
   * - Hardware E-stop
     - Operator can stop both arms without the GPU node.
     - Live test before every session
   * - Workspace and action limits
     - Clip Cartesian/TCP commands and enforce velocity, force, and joint limits.
     - Config review and dry run
   * - Camera freshness
     - Reject stale or missing frames; do not reuse an old positive label.
     - Timestamp and fault logs
   * - Invalid output
     - Map invalid generations to ``invalid_reward: 0.0`` and log the text.
     - Parser metrics
   * - Unclear output
     - Start with ``unclear_reward: 0.0``.
     - Held-out calibration
   * - Reward outage
     - Freeze policy updates and fall back to a documented independent signal.
     - Fault-injection test
   * - Rollback
     - Keep the last known-good policy and reward checkpoint.
     - Restore test

Shadow-to-RL gate
-----------------

Promote only when all rows are green:

.. list-table::
   :header-rows: 1
   :widths: 34 42 24

   * - Gate
     - Check
     - Stop condition
   * - Data
     - No train/eval episode overlap and no unresolved camera faults.
     - Any leakage or missing required view
   * - Judge
     - Held-out metrics meet the zero-shot or LoRA acceptance target.
     - Invalid rate or positive recall fails
   * - Latency
     - p95 reward latency fits the selected ``input_interval``.
     - Queue growth or stale windows
   * - Shadow agreement
     - Operator and judge agree on a reviewed stream.
     - Repeated unsafe disagreement
   * - Policy
     - Dry-run actions, limits, reset, and checkpoint restore all pass.
     - Any uncontrolled motion

Reward shaping defaults
-----------------------

Use conservative scalar mappings during the pilot:

.. code-block:: yaml

   reward_parser_params:
     positive_reward: 1.0
     negative_reward: -0.2
     unclear_reward: 0.0
     invalid_reward: 0.0

Do not add a large ``gt_success_bonus`` until you have verified that the independent
success signal is correct for the Flexiv task. A large bonus can hide a weak trend
judge and make policy updates brittle.

Failure modes to test
---------------------

Inject one fault at a time: unplug a camera, delay frames, return an invalid model
string, stop the reward worker, disconnect the robot controller, and press the
operator E-stop. The expected result is a safe stop or documented fallback, never a
new exploratory action.

Record the outcome in the experiment manifest. A run without fault-injection results
is an observation session, not an RL acceptance test.

Final handoff
-------------

Before enabling policy updates, attach this handoff table to the run:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Record
     - Required content
   * - Checkpoints
     - Policy, reward, and last known-good checkpoint paths.
   * - Revisions
     - Prompt, camera-selection, and label-policy revisions.
   * - Evidence
     - Held-out metrics and shadow agreement summary.
   * - Session
     - Hardware limits and operator name.
   * - Recovery
     - Rollback command and verified restore result.

Then start with a short evaluation-only run, inspect ``env/reward`` and
``env/reward_model_output``, and increase rollout length only after the operator
signs off.
