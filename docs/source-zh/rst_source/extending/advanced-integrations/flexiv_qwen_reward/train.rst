LoRA 微调
=========

只有当基座模型没有通过 zero-shot 门槛时，才将 Qwen3-VL 微调为 Flexiv 趋势 judge。
保持基座模型路径和 reward LoRA 路径分离，便于比较和回滚。

安装 VLM 环境
------------

分别为 VLM 和机器人栈创建虚拟环境：

.. code-block:: bash

   bash requirements/install.sh embodied \
       --model qwen3_vl --env maniskill_libero \
       --venv .venv-qwentrend

   bash requirements/install.sh embodied \
       --env bi_flexiv \
       --venv .venv-bi-flexiv

这两条命令分别创建 Qwen3-VL reward 环境和 Flexiv 机器人环境。当前 ``qwen3_vl`` installer 的环境参数只接受
``maniskill_libero`` 或 ``libero``，还不能直接与 ``bi_flexiv`` 组合。在安装逻辑扩展前，请保持两个环境分离。
OpenPI 等策略模型应安装在策略节点使用的环境中。

准备 SFT 数据集
--------------

标准数据集名称是 ``qwentrend_progress_sft``。将数据根目录指向
:doc:`数据与标签 <dataset>` 中已经验收的 train/eval manifest：

.. code-block:: bash

   export DUALVIEW_SFT_DATA_ROOT=/path/to/processed_flexiv_qwentrend

确认该目录包含 ``train/segments.jsonl`` 和 ``eval/segments.jsonl``。现有数据加载器会相对于该根目录解析逐样本视频路径，
并要求每个视频包含 5 帧。

以现有配置为起点：

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

启动 SFT
--------

使用仓库已有的 VLM SFT 入口和 QwenTrend 配置：

.. code-block:: bash

   bash examples/sft/run_vlm_sft.sh qwen3vl_sft_qwentrend

该命令会加载 Qwen3-VL processor，训练配置的 LoRA adapter，并将 checkpoint 写入 SFT 输出目录。
优化器、FSDP 和样本重加权参数以 ``qwen3vl_sft_qwentrend`` 配置为准。

部署前评测
----------

在同一个按 episode 隔离、由人工标注的评测集上比较 LoRA checkpoint 与基座模型。
报告 Macro F1、positive recall、unclear recall、无效输出率和 p95 延迟。不能只根据训练 loss 选择 checkpoint。

优先做最小修改来修复错误。例如模型把“已经抓住”误判为“已经完成”时，加入夹爪姿态相同但任务进度不同的反例，
不要直接增大模型规模。

部署契约
--------

在在线 reward 配置中通过 ``reward.model.lora_path`` 传入训练好的 adapter。在线 wrapper 仍然使用
``model_type: history_vlm``；当前 RLinf 树中没有注册 ``buffered_vlm``。

每个 checkpoint 旁都保存 manifest，记录基座模型 revision、prompt revision、相机选择、标签策略、数据集 hash 和评测指标。
缺少这些元数据的 checkpoint 不能部署到真机。

下一步
------

阅读 :doc:`在线 RL 接入 <online_rl>`，先以 shadow mode 加载模型。
