LoRA Fine-Tuning
================

Fine-tune Qwen3-VL as a Flexiv trend judge only after the base model fails the
zero-shot gate. Keep the base model path and the reward LoRA path separate so you
can compare and roll back both variants.

Install the VLM stack
---------------------

Install the VLM and robot stacks in separate virtual environments:

.. code-block:: bash

   bash requirements/install.sh embodied \
       --model qwen3_vl --env maniskill_libero \
       --venv .venv-qwentrend

   bash requirements/install.sh embodied \
       --env bi_flexiv \
       --venv .venv-bi-flexiv

What this does: creates one Qwen3-VL reward environment and one Flexiv robot
environment. The current ``qwen3_vl`` installer accepts ``maniskill_libero`` or
``libero`` as its environment argument; it does not yet compose directly with
``bi_flexiv``. Keep the environments separate until install support is extended.
Install the policy model, such as OpenPI, in the environment used by the policy node.

Prepare the SFT dataset
-----------------------

The canonical dataset name is ``qwentrend_progress_sft``. Set the data root to the
accepted train/eval manifests from :doc:`Data and Labels <dataset>`:

.. code-block:: bash

   export DUALVIEW_SFT_DATA_ROOT=/path/to/processed_flexiv_qwentrend

Check that the root contains ``train/segments.jsonl`` and ``eval/segments.jsonl``.
The existing dataset loader resolves the per-sample video paths relative to this
root and expects five frames per video.

Use the existing config as the starting point:

.. code-block:: yaml

   data:
     type: vlm
     dataset_name: "qwentrend_progress_sft"
     train_data_paths: "${oc.env:DUALVIEW_SFT_DATA_ROOT}/train/segments.jsonl"
     val_data_paths: "${oc.env:DUALVIEW_SFT_DATA_ROOT}/eval/segments.jsonl"
     video_root: "${oc.env:DUALVIEW_SFT_DATA_ROOT}"
     data_root: "${oc.env:DUALVIEW_SFT_DATA_ROOT}"
     video_nframes: 5

   actor:
     model:
       model_type: qwen3_vl
       model_path: /path/to/Qwen3-VL-4B-Instruct
       is_lora: true
       lora_rank: 16

Start SFT
---------

Run the repository's VLM SFT entry point with the QwenTrend config:

.. code-block:: bash

   bash examples/sft/run_vlm_sft.sh qwen3vl_sft_qwentrend

What this does: loads the Qwen3-VL processor, trains the configured LoRA adapters,
and writes checkpoints under the configured SFT output directory. Keep the
``qwen3vl_sft_qwentrend`` config as the source of truth for optimizer, FSDP, and
sample-reweighting settings.

Evaluate before deployment
--------------------------

Compare the LoRA checkpoint with the base model on the same episode-disjoint,
human-labeled evaluation set. Report macro F1, positive recall, unclear recall,
invalid-output rate, and p95 latency. Do not select a checkpoint by training loss
alone.

Prefer the smallest change that fixes the observed error. For example, if the model
confuses “grasped” with “completed,” add counterexamples with the same gripper pose
at different task progress points instead of increasing the model size.

Deployment contract
-------------------

Pass the trained adapter to the online reward config with ``reward.model.lora_path``.
The online wrapper remains ``model_type: history_vlm``; ``buffered_vlm`` is not a
registered model type in the current RLinf tree.

Keep a manifest next to every checkpoint containing the base model revision, prompt
revision, camera selection, label policy, dataset hash, and evaluation metrics. A
checkpoint without this metadata is not ready for physical deployment.

Next step
---------

Follow :doc:`Online RL Integration <online_rl>` to load the model in shadow mode.
