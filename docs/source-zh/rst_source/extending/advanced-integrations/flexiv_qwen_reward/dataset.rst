数据与标签
==========

构建按 episode 隔离的数据集，并让标签描述“进展”，而不只是最后一帧是否成功。
这样可以避免 reward model 学到“夹爪已闭合”或“物体已经到达目标”等捷径。

原始 episode 契约
-----------------

在记录 Flexiv rollout 的环境上开启 ``env.data_collection``。采集器会为每个 episode 写入一个 ``.pkl`` 文件。
请保留以下字段：

.. list-table::
   :header-rows: 1
   :widths: 28 52 20

   * - 字段
     - 含义
     - 要求
   * - ``observations``
     - 有序观测，包含 ``main_images`` 和选定的额外视角。
     - 必须
   * - ``task`` 或 ``task_description``
     - Prompt 中使用的任务指令。
     - 必须
   * - ``rewards`` / ``gae``
     - 供现有自动预处理器使用的进展信号。
     - 仅 score 标签需要
   * - ``episode_id``
     - 用于划分和审计的稳定 episode 标识。
     - 必须
   * - 时间戳与安全事件
     - 相机新鲜度、人工介入、停止或故障信息。
     - 强烈建议

Flexiv 环境配置已经在 ``examples/embodiment/config/env/realworld_bi_flexiv_tcp.yaml`` 中提供任务文本和相机设置。
采集前必须替换所有 ``REPLACE_WITH_*`` 值；如果相机序列号能够识别具体设备，不要将其放入公开 manifest。

标签来源
--------

按以下优先级使用标签：

.. list-table::
   :header-rows: 1
   :widths: 24 48 28

   * - 来源
     - 用途
     - 规则
   * - 人工标注
     - 初始 zero-shot 评测和真实世界的歧义样本。
     - 保存标注人、置信度和备注。
   * - 任务状态或仪器信号
     - 当信号独立于 Qwen3-VL 时生成大规模训练标签。
     - 给规则版本并记录来源。
   * - ``rewards`` / ``gae`` 差值
     - 供现有 QwenTrend 预处理器使用。
     - 只有 score 有意义且不是同一个 VLM 生成时才使用。
   * - Qwen3-VL 伪标签
     - 排序或主动学习建议。
     - 不能作为唯一真值。

当前 ``preprocess_qwentrend_reward_dataset.py`` 根据窗口的 score 差值生成标签，小差值标为 ``unclear``，
并可以反转窗口合成负样本。该行为适用于仿真或有仪器信号的任务；当 Flexiv 没有独立 reward 时，不能替代人工标签。

窗口与划分策略
--------------

从 ``window_size: 5`` 和同步帧开始。一次对比中保持采样间隔固定。先按 episode 划分，再在每个 split 中平衡标签。
不要把反转增强样本和其原始 episode 放入不同 split。

对于基于 score 的数据，现有命令是：

.. code-block:: bash

   python examples/reward/preprocess_qwentrend_reward_dataset.py \
       --raw-data-path logs/<run>/collected_data \
       --output-dir logs/<run>/processed_qwentrend_reward_data \
       --window-size 5 \
       --stride 1

该命令会切出双视角窗口，写入逐样本 pickle 文件和 JSONL manifest，并按 episode 生成 ``train`` / ``eval`` split。
在确认 ``rewards`` 或 ``gae`` 有独立且合理的含义前，不要对 Flexiv 数据执行此命令。

人工 manifest 适配器
-------------------

LoRA 训练前需要添加一个人工标注适配器。其输出应兼容 ``qwentrend_progress_sft``，
每条 JSONL 记录指向一个 pickle payload：

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

对应 pickle payload 必须包含 ``main_frames`` 和 ``extra_view_frames`` 数组。
让 ``answer`` 与目标标签完全相同。将标注元数据放在 prompt 外部，便于修改 prompt 而不重写标签。

数据集验收
----------

训练前检查每个样本是否有每个视角各 5 帧、有效 RGB shape、一条任务文本、三类标签之一和 episode 标识。
报告标签数量、相机缺失数量、重复 sample ID 和安全事件数量。除非有意将其作为 ``unclear`` 或失败样本，
否则应拒绝含相机故障或人工停止的 split。

下一步
------

将 ``DUALVIEW_SFT_DATA_ROOT`` 指向通过验收的数据集，然后阅读
:doc:`LoRA 微调 <train>`。
