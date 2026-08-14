Bi-Flexiv Record 的离线 RECAP
==============================

使用 Bi-Flexiv 的 LeRobot record 离线训练新的 π₀.₅ checkpoint，不让 RLinf
连接机械臂。RECAP 依次生成轨迹 return、训练本地 SigLIP2 + Gemma3 价值模型、
计算逐帧 advantage，最后使用 classifier-free guidance 优化 OpenPI 策略。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 算法
      :text-align: center

      RECAP + CFG

   .. grid-item-card:: 模型
      :text-align: center

      RECAP value VLM · π₀.₅

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      Bi-Flexiv · LeRobot v3

   .. grid-item-card:: 训练
      :text-align: center

      离线 · 四阶段

| **你将完成：** 验证 record → 计算 return → 训练 value VLM → 计算 advantage → 训练 π₀.₅。
| **前置条件：** OpenPI 环境 · SigLIP2 checkpoint · Gemma3 checkpoint · π₀.₅ PyTorch checkpoint。

LeRobot 版本
------------

Flexiv recorder 输出的是 LeRobot v3 file-based 格式。如果 OpenPI 环境中是
``lerobot==0.3.3``，训练前需要执行一次定向覆盖，安装锁定的 v3 reader：

.. code:: bash

   uv pip install --force-reinstall --no-deps \
     "lerobot @ git+https://github.com/huggingface/lerobot.git@v0.4.4"

检查当前环境：

.. code:: bash

   python -c "from importlib.metadata import version; print(version('lerobot'))"

LeRobot v3 record 应输出 ``0.4.4``。RLinf 现在会使用显式 ``root`` 打开本地
数据集；版本不匹配时会直接报告版本错误，不再回退访问不存在的 Hugging Face
Hub 仓库。

准备数据集
----------

Flexiv recorder 输出已经是 LeRobot v3。解压并验证：

.. code:: bash

   python examples/offline_rl/prepare_flexiv_recap.py \
     --input-zip /path/to/stack-cubes-flexiv.zip \
     --output-dir /data/flexiv-recap

使用命令最后输出的数据集路径。这个目录应直接包含 ``data/``、``meta/`` 和
``videos/``。

.. warning::

   当前 record 只有一个 episode。下面的命令把它当作成功的专家演示
   （``type=sft``）。这足够验证链路，但要让策略明显提升，通常需要采集多条
   成功和失败轨迹，覆盖不同状态与动作。

运行
----

先设置路径：

.. code:: bash

   export FLEXIV_DATA=/data/flexiv-recap/stack-cubes-flexiv-0811
   export SIGLIP_PATH=/models/siglip2-so400m-patch14-224
   export GEMMA_PATH=/models/gemma-3-270m
   export PI05_PATH=/models/pi05_base_pytorch

Step 1 生成 ``meta/returns_success.parquet``：

.. code:: bash

   bash examples/offline_rl/advantage_labeling/recap/process/run_compute_returns.sh \
     recap_bi_flexiv_compute_returns \
     data.train_data_paths.0.dataset_path=$FLEXIV_DATA

Step 2 训练本地 RECAP value VLM：

.. code:: bash

   bash examples/offline_rl/advantage_labeling/recap/run_value_sft.sh \
     recap_bi_flexiv_value_model_sft \
     data.train_data_paths.0.dataset_path=$FLEXIV_DATA \
     actor.model.siglip_path=$SIGLIP_PATH \
     actor.model.gemma3_path=$GEMMA_PATH \
     actor.model.tokenizer_path=$GEMMA_PATH

把 ``VALUE_CKPT`` 指向训练得到的 ``global_step_N/actor``。Step 3 使用 VLM
生成 ``meta/advantages_recap.parquet``：

.. code:: bash

   export VALUE_CKPT=/path/to/value_sft/checkpoints/global_step_N/actor

   bash examples/offline_rl/advantage_labeling/recap/process/run_compute_advantages.sh \
     recap_bi_flexiv_compute_advantages --nproc 1 \
     data.train_data_paths.0.dataset_path=$FLEXIV_DATA \
     advantage.value_checkpoint=$VALUE_CKPT \
     advantage.model.siglip_path=$SIGLIP_PATH \
     advantage.model.gemma3_path=$GEMMA_PATH \
     advantage.model.tokenizer_path=$GEMMA_PATH

Step 4 根据 advantage 标签优化 π₀.₅：

.. code:: bash

   bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh \
     cfg_rl_bi_flexiv \
     data.train_data_paths.0.dataset_path=$FLEXIV_DATA \
     actor.model.model_path=$PI05_PATH

更新后的策略位于
``logs/cfg_rl/cfg_rl_bi_flexiv-*/checkpoints/global_step_N/actor``。在 Flexiv
原有的 OpenPI 推理进程中加载这个 actor checkpoint；部署推理不需要 RLinf。

Reward 的工作方式
------------------

RECAP 的本地 VLM 是训练出来的价值函数，不是 zero-shot reward 裁判。Step 1
根据轨迹是否成功和距离结束的步数初始化 reward/return。Step 2 让 VLM 根据相机
图像和任务文本预测 return。Step 3 将 VLM 预测与 N-step reward 合成 advantage。
Step 4 根据正负 advantage 标签更新 π₀.₅。
