Zero-Shot 验证
=============

在创建 reward 数据集或修改策略前，先测量原版 Qwen3-VL 能否判断 Flexiv 的任务进展。
此阶段只做离线评测和 shadow mode。

输入契约
--------

每个样本由一条任务指令和两段同步视频组成：

.. list-table::
   :header-rows: 1
   :widths: 28 52 20

   * - 字段
     - 必须内容
     - RLinf 来源
   * - ``task``
     - 一句祈使句，例如 ``Place the bottle in the carton.``
     - ``override_cfg.task_description``
   * - ``main_images``
     - 固定头部相机的 5 帧 RGB 图像。
     - ``RealWorldEnv`` 观测
   * - ``extra_view_images``
     - 固定腕部或第三人称相机的 5 帧 RGB 图像。
     - Flexiv 相机适配器
   * - ``label``
     - 人工参考标签：``positive``、``negative`` 或 ``unclear``。
     - 标注 manifest

当前 ``realworld_bi_flexiv_tcp.yaml`` 会通过 ``extra_view_images`` 暴露头部相机和两路腕部相机。
在线 QwenTrend builder 消费的是两段视频，而不是任意数量的腕部视角。请在 Flexiv 适配器中固定选取一路
腕部视角，或新增 ``flexiv_qwentrend_input_builder``。采集、评测和 RL 阶段不能静默改变相机顺序。

Prompt 与输出
-------------

离线评测与在线推理必须使用同一个 prompt：

.. code-block:: text

   You are currently performing the task: <task>.
   You are given two synchronized 5-frame videos from different camera views
   (main view and third-person view) of the same robot action window. Judge whether
   the action trend is positive, negative, or unclear. Answer with exactly one word:
   positive, negative, or unclear.

现有 ``qwentrend_reward_parser`` 支持这些标签，也支持从带 ``trend`` 字段的 JSON 对象中提取标签。
标注数据中请保留精确标签，便于发现 parser 错误。

评测集
------

至少采集几百个窗口，覆盖正常进展、可恢复错误、停滞和相机遮挡。操作员为每个窗口标注，
并对有代表性的子集进行双人复核。必须按 episode 划分，而不是按窗口划分，避免相邻帧泄漏到 train 和 eval。

记录以下指标：

.. list-table::
   :header-rows: 1
   :widths: 34 48 18

   * - 指标
     - 作用
     - 建议门槛
   * - 三类标签 Macro F1
     - 防止多数类 ``unclear`` 让 judge 看起来虚高。
     - ``>= 0.85``
   * - Positive recall
     - 确保真正的进展不会被丢弃。
     - ``>= 0.90``
   * - 无效输出率
     - 暴露 prompt、tokenizer 或 parser 失败。
     - ``<= 1%``
   * - p95 推理延迟
     - 决定控制循环中 ``input_interval`` 是否足够安全。
     - 在目标硬件测量

这些是试验阶段的起始门槛，不是 RLinf 的保证。任务和操作员数量增加后应继续收紧。

实现缺口
--------

在把本流程做成命令行 recipe 前，先添加一个独立离线评测器。建议接口如下：

.. code-block:: text

   examples/reward/eval_flexiv_qwentrend_zero_shot.py
       --manifest <jsonl>
       --model-path <Qwen3-VL directory>
       --output <metrics.json>

评测器应保存原始生成文本、解析标签、延迟、相机元数据和混淆矩阵。该工具必须与 RL runner 分离，
避免错误的 judge 驱动机械臂。

下一步
------

如果基座模型通过留出集门槛，保持 ``lora_path`` 未设置，并进入
:doc:`在线 RL 接入 <online_rl>` 的 shadow mode。否则先阅读
:doc:`数据与标签 <dataset>` 和 :doc:`LoRA 微调 <train>`。
