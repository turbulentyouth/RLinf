在线 RL 接入
===========

通过 RLinf 的 reward worker 接入 Flexiv 监测器。本地 Qwen3-VL checkpoint 使用
``history_vlm``；云端 endpoint 使用 :doc:`云端 Qwen API <cloud_api>`。在目标机器人节点
测量标签、延迟和失败行为前，保持 monitor 处于 shadow mode。

观测映射
--------

``RealWorldEnv`` 将头部相机暴露为 ``main_images``，其余相机暴露为 ``extra_view_images``，任务文本暴露为
``task_descriptions``。QwenTrend 要求两段同步的 5 帧视频。如果默认额外视角包含两路腕部相机，
请在 input builder 中确定性地选取一路并记录选择。

Reward 配置
-----------

本地推理可以以现有 reward model 字段为起点：

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

   只有当 Flexiv 相机适配器向 ``qwentrend_input_builder`` 提供恰好一路额外视角后，
   此配置块才可以使用。当前观测包含两路腕部相机时，应实现并选择 Flexiv 专用 input builder。
   Shadow run 中保持 ``reward_weight: 0.0``。

字段作用：

.. list-table::
   :header-rows: 1
   :widths: 30 52 18

   * - 字段
     - 作用
     - Flexiv 起点
   * - ``history_size`` / ``min_history_size``
     - 只有完整时间窗口才进行推理。
     - ``5`` / ``5``
   * - ``input_interval``
     - 每 N 个环境 step 运行一次 monitor。
     - 从 ``1`` 开始；根据延迟测试再增加
   * - ``infer_micro_batch_size``
     - 限制 reward 推理的 GPU 内存。
     - 单机器人从 ``1`` 开始
   * - ``reward_weight``
     - 缩放 learned reward 在最终 reward 中的权重。
     - shadow mode 从 ``0`` 开始
   * - ``env_reward_weight``
     - 保留独立环境 reward（如果存在）。
     - 有独立 reward 时保持 ``1``；否则记录为何设置为 ``0``

当前 EnvWorker 使用 ``env_reward_weight * env_reward + reward_weight * reward_model_output`` 合并外部 reward。
当 ``history_reward_assign: true`` 时，历史结果由 history manager 分配到对应窗口。为新 Flexiv 任务修改这些语义前，
请先阅读实现。

Shadow mode
-----------

首次在线运行不得让 monitor 改变动作、reset 或用于更新的 reward。添加只记录日志的路径，保存原始生成、解析标签、
标量映射、时间戳、相机 ID 和操作员判断。如果复用 RL runner，可将 ``reward_weight: 0.0``，同时保持 reward worker 启用，
这样仍能测试延迟和输出。

Flexiv 顶层配置
---------------

使用
``examples/embodiment/config/realworld_bi_flexiv_ppo_openpi_pi05_cloud_qwen.yaml``。
它包含以下内容：

.. list-table::
   :header-rows: 1
   :widths: 34 46 20

   * - 部分
     - 必须内容
     - 来源
   * - ``env.train`` / ``env.eval``
     - ``env_type: realworld`` 和 Flexiv 环境 include。
     - ``env/realworld_bi_flexiv_tcp.yaml``
   * - ``cluster``
     - 一个机器人节点组和一个或多个 GPU placement。
     - 异构集群指南
   * - ``actor`` / ``rollout``
     - Flexiv 所需的策略 checkpoint 和动作 shape。
     - 策略专用配置
   * - ``reward``
     - 上面的 QwenTrend 配置块。
     - 本页
   * - ``runner``
     - 先运行短时 eval-only 或 shadow 实验。
     - 现有 embodied runner

仓库配置默认使用 ``is_dummy: true``。dummy、硬件 shadow 和在线 RL 的启动命令见
:doc:`云端 Qwen API <cloud_api>`。

下一步
------

在设置非零 reward weight 或允许策略更新前，先阅读 :doc:`安全与验收 <safety>`。
