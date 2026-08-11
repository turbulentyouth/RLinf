安全与验收
==========

将 VLM 视为任务进展监测器，而不是急停控制器。Qwen3-VL 输出可能过时、格式错误，
或者看似合理但实际错误。硬件限制和人工介入必须独立于模型。

必须具备的安全措施
------------------

.. list-table::
   :header-rows: 1
   :widths: 34 46 20

   * - 安全措施
     - 要求
     - 证据
   * - 硬件急停
     - 操作员无需 GPU 节点即可停止两条机械臂。
     - 每次实验前现场测试
   * - 工作空间与动作限制
     - 裁剪笛卡尔/TCP 指令，并执行速度、力和关节限制。
     - 配置检查和 dry run
   * - 相机新鲜度
     - 拒绝过期或缺失帧；不能复用旧的 positive 标签。
     - 时间戳和故障日志
   * - 无效输出
     - 映射为 ``invalid_reward: 0.0`` 并记录原始文本。
     - Parser 指标
   * - Unclear 输出
     - 试验阶段设为 ``unclear_reward: 0.0``。
     - 留出集校准
   * - Reward 中断
     - 冻结策略更新，回退到有文档的独立信号。
     - 故障注入测试
   * - 回滚
     - 保留最后一个已知正常的策略和 reward checkpoint。
     - 恢复测试

Shadow 到 RL 的门槛
-------------------

只有所有行都通过才允许升级：

.. list-table::
   :header-rows: 1
   :widths: 34 42 24

   * - 门槛
     - 检查
     - 停止条件
   * - 数据
     - train/eval 没有 episode 重叠，也没有未解决的相机故障。
     - 发生泄漏或缺少必需视角
   * - Judge
     - 留出集指标达到 zero-shot 或 LoRA 验收目标。
     - 无效率或 positive recall 不达标
   * - 延迟
     - p95 reward 延迟适配选定的 ``input_interval``。
     - 队列增长或窗口过时
   * - Shadow 一致性
     - 在复核流中操作员与 judge 保持一致。
     - 重复出现不安全分歧
   * - 策略
     - dry-run 动作、限制、reset 和 checkpoint 恢复全部通过。
     - 出现任何不可控运动

Reward shaping 默认值
---------------------

试验阶段使用保守的标量映射：

.. code-block:: yaml

   reward_parser_params:
     positive_reward: 1.0
     negative_reward: -0.2
     unclear_reward: 0.0
     invalid_reward: 0.0

在确认独立 success signal 对 Flexiv 任务正确前，不要加入很大的 ``gt_success_bonus``。
大 bonus 会掩盖趋势 judge 的弱点，并让策略更新变得脆弱。

必须测试的故障
--------------

一次注入一个故障：拔掉相机、延迟帧、返回无效模型字符串、停止 reward worker、断开机器人控制器，
以及按下操作员急停。预期结果应是安全停止或有文档的回退，不能产生新的探索动作。

将结果写入实验 manifest。没有故障注入结果的运行只能算观测会话，不能算 RL 验收测试。

最终交接
--------

允许策略更新前，将以下交接表附在运行记录中：

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 记录
     - 必须内容
   * - Checkpoint
     - 策略、reward 和最后一个已知正常的 checkpoint 路径。
   * - Revision
     - Prompt、相机选择和标签策略 revision。
   * - 证据
     - 留出集指标和 shadow 一致性总结。
   * - 会话
     - 硬件限制与操作员姓名。
   * - 恢复
     - 回滚命令和已验证的恢复结果。

随后先运行短时 evaluation-only，检查 ``env/reward`` 和 ``env/reward_model_output``，
只有操作员签字后才增加 rollout 长度。
