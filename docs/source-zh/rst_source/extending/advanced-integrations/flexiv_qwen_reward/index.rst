Flexiv QwenTrend 奖励流程
=========================

使用本合集将 Flexiv 相机流转换为可监测、可控的学习信号。先做离线 zero-shot
判断；只有当基座模型在留出数据上不够可靠时，再微调 Qwen3-VL。

.. important::

   这是 Flexiv 集成的实现蓝图。RLinf 已包含 ``bi_flexiv`` 环境、本地和云端 history
   reward worker、Flexiv OpenPI pi0.5 顶层配置、QwenTrend 输入与解析工具，以及
   QwenTrend SFT 数据集；但目前还没有独立 zero-shot 评测 CLI 和人工标注适配器。
   真机在线 RL 依赖该 judge 前，需要补齐这两个组件。

架构
----

将策略和监测器分开：

.. code-block:: text

   Flexiv 相机 + 任务文本
            |
            v
   头部视角 + 选定腕部视角，5 帧
            |
            v
   Qwen3-VL（可选 reward LoRA）
            |
            v
   positive / negative / unclear
            |
            v
   标量 reward -> RLinf EnvWorker -> 更新策略

Qwen3-VL 是通用 VLM，不是厂商提供的机器人 reward model。Flexiv 专用部分来自
任务 prompt、相机适配器、标签策略，以及用 Flexiv 数据训练得到的可选 LoRA checkpoint。

分阶段上线
----------

.. list-table::
   :header-rows: 1
   :widths: 14 28 38 20

   * - 阶段
     - 目标
     - 必须具备的证据
     - 对策略的影响
   * - 0
     - 硬件与相机 dry run
     - ``main_images``、选定腕部视角、任务文本和时间戳稳定。
     - 无
   * - 1
     - Zero-shot judge
     - 人工标注的 5 帧窗口、混淆矩阵、无效输出率和延迟。
     - 仅 shadow
   * - 2
     - Reward 数据集与 LoRA
     - 按 episode 隔离的 train/eval 划分，以及与 zero-shot 的留出集对比。
     - 仅 shadow
   * - 3
     - 在线 reward shadow mode
     - reward 日志与操作员判断一致，且不改变 reset 或动作。
     - 无
   * - 4
     - 保守在线 RL
     - 通过安全检查，并准备好可回滚 checkpoint。
     - 受限更新

选择阶段：

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: Zero-Shot 验证
      :link: zero_shot
      :link-type: doc

      在训练前测试原版 Qwen3-VL。

   .. grid-item-card:: 数据与标签
      :link: dataset
      :link-type: doc

      定义 Flexiv episode 契约并避免标签泄漏。

   .. grid-item-card:: LoRA 微调
      :link: train
      :link-type: doc

      只有 zero-shot 未通过门槛时才微调 Qwen3-VL。

   .. grid-item-card:: 在线 RL 接入
      :link: online_rl
      :link-type: doc

      先以 shadow mode 将 ``history_vlm`` 接入 reward channel。

   .. grid-item-card:: 云端 Qwen API
      :link: cloud_api
      :link-type: doc

      从 reward worker 调用可配置的 Qwen-compatible endpoint。

   .. grid-item-card:: 安全与验收
      :link: safety
      :link-type: doc

      用明确的停止和回滚条件管理真机实验。

当前 RLinf 组件
--------------

.. list-table::
   :header-rows: 1
   :widths: 40 36 24

   * - 组件
     - 路径
     - 状态
   * - Flexiv 环境
     - ``examples/embodiment/config/env/realworld_bi_flexiv_tcp.yaml``
     - 已有
   * - 观测 wrapper
     - ``rlinf/envs/realworld/realworld_env.py``
     - 已有
   * - QwenTrend 预处理
     - ``examples/reward/preprocess_qwentrend_reward_dataset.py``
     - 已有；要求基于 score 生成标签
   * - QwenTrend SFT 配置
     - ``examples/sft/config/qwen3vl_sft_qwentrend.yaml``
     - 已有
   * - 在线 reward model
     - ``model_type: history_vlm``
     - 本地 VLM 权重可用；云端 endpoint 使用 ``dashscope_history_vlm``
   * - Flexiv 顶层 RL 配置
     - ``realworld_bi_flexiv_ppo_openpi_pi05_cloud_qwen.yaml``
     - 已有；默认使用 dummy mode

.. toctree::
   :hidden:

   Zero-Shot 验证 <zero_shot>
   数据与标签 <dataset>
   LoRA 微调 <train>
   在线 RL 接入 <online_rl>
   云端 Qwen API <cloud_api>
   安全与验收 <safety>
