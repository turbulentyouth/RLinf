云端 Qwen API 奖励
==================

让奖励判断在支持 Qwen 的云端 endpoint 上运行，无需下载 VLM 权重。RLinf 将策略
rollout 与云端监测器分开：监测器接收一段短图像历史，返回 ``positive``、
``negative`` 或 ``unclear``，再由现有 reward worker 将标签映射为标量。

API 在哪里调用
---------------

唯一的外发请求位于
``rlinf/models/embodiment/reward/dashscope_vlm_reward_model.py`` 的
``_call_api``。运行时链路是：

.. code-block:: text

   EnvWorker.get_reward_model_output
       -> RewardWorker.compute_image_rewards
       -> DashScopeHistoryVLMRewardModel.compute_reward
       -> DashScopeHistoryVLMRewardModel._call_api
       -> 服务商 /v1/chat/completions endpoint
       -> qwentrend_reward_parser
       -> 标量 reward
       -> EnvWorker 最终 reward

该模型以 ``dashscope_history_vlm`` 注册在
``rlinf/models/embodiment/reward/__init__.py``。HTTP 客户端使用 Python 标准库，
因此不需要额外 SDK。请将 ``api_url`` 和 ``api_model`` 设置为账号可用的值；示例
使用 DashScope 的 OpenAI-compatible endpoint，但两者都可以配置。

配置凭证
--------

只在进程环境中导出 API key。不要把 key 写进 YAML、checkpoint 或实验 manifest：

.. code-block:: bash

   export DASHSCOPE_API_KEY='<your-key>'
   export QWEN_API_MODEL='<provider-supported-qwen-vl-model>'

这两个变量会在 reward worker 启动时读取。模型名保留为环境变量，是因为云端可用
模型和权限取决于服务商账号。

使用云端 reward 配置
--------------------

先运行单机 Flexiv 顶层配置。仓库中的默认值为 ``is_dummy: true``，因此不会连接
机械臂或相机：

.. code-block:: bash

   export FLEXIV_PI05_MODEL_PATH=/path/to/pi05_bi_flexiv_checkpoint
   export FLEXIV_TASK_DESCRIPTION='Put the object into the box.'
   export DASHSCOPE_API_KEY='<your-key>'
   export QWEN_API_MODEL='<provider-supported-qwen-vl-model>'

   bash examples/embodiment/run_realworld.sh \
       realworld_bi_flexiv_ppo_openpi_pi05_cloud_qwen

该命令加载本地 OpenPI pi0.5 策略，创建全零的 Flexiv dummy observation，为有效的
5 帧窗口调用云端 judge，计算 GAE，并执行 PPO actor 更新。它验证完整软件链路，但不会
驱动硬件。配置文件是
``examples/embodiment/config/realworld_bi_flexiv_ppo_openpi_pi05_cloud_qwen.yaml``。

dummy 运行通过后，再进行硬件 shadow：

.. code-block:: bash

   export EMBODIED_PATH="$PWD/examples/embodiment"
   export PYTHONPATH="$PWD:${PYTHONPATH}"

   python examples/embodiment/train_embodied_agent.py \
       --config-path "$EMBODIED_PATH/config" \
       --config-name realworld_bi_flexiv_ppo_openpi_pi05_cloud_qwen \
       env.train.override_cfg.is_dummy=false \
       env.train.override_cfg.dry_run=true \
       reward.reward_weight=0.0

该命令会连接并 reset Flexiv，读取真实相机，但在策略动作到达机械臂前拦截。API 输出
记录为 ``env/reward_model_output``；由于 ``reward.reward_weight=0.0``，它不会影响 PPO。

.. warning::

   只有显式设置 ``is_dummy=false``、``dry_run=false`` 和非零
   ``reward.reward_weight`` 才会进入在线 RL。请先验证硬件限制、相机顺序、API 延迟和
   checkpoint 回滚。

所有安全门槛通过后，先只执行一次 PPO 更新：

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-path "$EMBODIED_PATH/config" \
       --config-name realworld_bi_flexiv_ppo_openpi_pi05_cloud_qwen \
       env.train.override_cfg.is_dummy=false \
       env.train.override_cfg.dry_run=false \
       reward.reward_weight=1.0 \
       runner.max_steps=1

该命令会把策略动作发送给 Flexiv，并使用 Qwen reward 完成一次 actor 更新。操作员必须
守在硬件急停旁。

.. code-block:: yaml

   reward:
     use_reward_model: true
     reward_mode: history_buffer
     history_reward_assign: true
     reward_weight: 0.0       # shadow mode
     env_reward_weight: 1.0
     model:
       model_type: dashscope_history_vlm
       api_key_env: DASHSCOPE_API_KEY
       api_url: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
       api_model: ${oc.env:QWEN_API_MODEL}
       api_timeout: 30.0
       api_max_retries: 2
       api_max_tokens: 16
       api_temperature: 0.0
       extra_view_index: 0
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
           input_interval: 5
           history_keys: [main_images, extra_view_images]
           input_on_done: false

将 ``extra_view_index`` 设置为离线验证过的腕部或第三人称相机。少于 5 帧的历史窗口
使用 ``interval_reward``，不会发出 API 请求。

最终 reward 如何产生
---------------------

当前 Flexiv 环境的 ``BiFlexivDualEnv.step()`` 返回 ``0.0``，``RealWorldEnv`` 将其
直接作为 environment reward。云端链路则按配置独立映射 Qwen 标签：

.. code-block:: text

   positive ->  1.0
   negative -> -0.2
   unclear  ->  0.0
   invalid  ->  0.0

``EnvWorker`` 使用下面的公式合并两路 reward：

.. code-block:: text

   final_reward = env_reward_weight * env_reward
                + reward_weight * qwen_reward

Flexiv 顶层配置设置 ``env_reward_weight: 0.0`` 和 ``reward_weight: 1.0``，因此 Qwen
是实际学习信号。启用 ``history_reward_assign: true`` 后，一次判断会分配给当前 step
以及有效 5 帧窗口内的前序 step。没有有效窗口的 step 使用
``interval_reward: 0.0``。

请求格式与响应解析
------------------

``_call_api`` 发送一个 JSON 请求。user message 包含任务 prompt 和
``data:image/jpeg;base64,...`` 格式的 JPEG 图像块。它要求 OpenAI-compatible 响应的
第一个 choice 包含 ``message.content``；content 可以是普通文本或文本块。随后由
``qwentrend_reward_parser`` 找到最后一个标签，并映射为 YAML 中配置的标量。

.. note::

   该集成使用可配置的 endpoint 兼容层，而不是绑定某个服务商 SDK。如果你的 endpoint
   使用不同的请求或响应 schema，只需修改 ``_call_api`` 和
   ``_extract_message_text``，``compute_reward`` 无需改变。

延迟、成本与故障行为
--------------------

云端推理在 reward worker 中同步执行。一个环境的一段 5 帧双视角窗口产生一次请求；
多个环境则每个环境一次请求。增大 ``input_interval``，并保持 ``api_max_retries`` 足够小，
避免请求队列超过控制周期。

先保持 ``reward_weight: 0.0``，检查请求延迟、解析器无效率和操作员一致性。重试仍失败时，
worker 抛出 ``RuntimeError``，让训练停止，而不是静默使用未知 reward。启用真机在线 RL
前，请先实现并验证独立的 fallback。

下一步
------

完成 shadow 验证后，阅读 :doc:`安全与验收 <safety>`。通过验收后，才在 Flexiv 专用配置中
设置非零 ``reward_weight``。
