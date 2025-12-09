**LLMob 框架与运行**

- 架构概览
  - 核心实体：`Person` 持有训练/测试轨迹、检索器与 LLM 句柄（`engine/agent.py:10-26`）。
  - 生成入口：`mob_gen(person, mode=0, scenario_tag, fast)` 执行动机生成与计划生成，写入结果（`engine/trajectory_generate.py:9-22, 85-90`）。
  - 检索机制：
    - 传统检索器：按时间/相似度从历史选择 `demo`（`engine/agent.py:28-32`）。
    - 意图检索器：用 VIMN 的意图向量从历史 Top‑K 中选更相关 `demo`（`engine/agent.py:33-48`）。
  - Prompt 流程：
    - 动机提示词：将 `attribute + demo + consecutive_past_days` 组合，生成较长期动机（`engine/trajectory_generate.py:30-55`）。
    - 计划提示词：将动机、日期、区域、weekday、demo 聚合，生成当天活动计划并写入 `results.pkl`（`engine/trajectory_generate.py:56-90`）。

- 运行逻辑
  - 单人生成命令：`python generate.py --dataset 2019 --mode 0 --id <id> [--fast] [--use_intent --intent_ckpt ...]`
    - `--fast` 只生成 1 个测试日，适合冒烟。
    - 启用意图检索时，`demo` 优先取意图检索 Top‑1，否则回退传统检索（`engine/trajectory_generate.py:33-42`）。
  - 输出位置：
    - 生成：`./result/<scenario_tag>/generated/llm_l/<id>/results.pkl`
    - 真值：`./result/<scenario_tag>/ground_truth/llm_l/<id>/results.pkl`

**VIMN 集成方法**

- 功能与接口
  - VIMN 输入：历史签到序列 `(poi_id, hour, category)`，从 `"Activities at YYYY-MM-DD: ..."` 文本解析并矢量化。
  - 编码器：三路嵌入拼接，经 Transformer 编码，取最后步隐藏态为“意图向量”。
  - 用法：以“意图检索器”的形式集成，不改变核心生成，仅替换 `demo`。
- 在 LLMob 中启用
  - 初始化：`Person.init_intent_retriever(ckpt_path=...)` 加载 `vimn_lite.pt` 并构建检索器（`engine/agent.py:33-48`）。
  - CLI 开关：`--use_intent --intent_ckpt ./engine/experimental/checkpoints/vimn_lite.pt`（`generate.py` 添加的参数）。
  - 生成流程的接入点：在 `mode=0` 下先取传统检索候选，再用意图检索 Top‑1 替换 `demo`（`engine/trajectory_generate.py:33-42`）。
- 训练与权重管理
  - 预训练脚本：`scripts/train_vimn.py`，支持按参数后缀保存权重，便于实验对照。
  - 权重位置：`./engine/experimental/checkpoints/vimn_lite.pt` 或带参数后缀的同目录文件。

**实验设计与结果**

- 设计目标
  - 对比“基线（传统检索）”与“VIMN 集成（意图检索替换 demo）”在：
    - 时间一致性（首事件小时 ±2h 命中率、时间 MAE）
    - 类别一致性（逐位匹配率、类别集合 Jaccard）
    - 稳定性（JSON 可解析率）
  - 方法参考 Mobility‑LLM 的两阶段思想：从历史识别模式与动机，再进行动机驱动生成；论文强调时间维度上的对齐表现更强，类别/地点严格匹配较难。[1]

- 运行脚本
  - 对比脚本：`scripts/run_intent_experiment.py`
    - 基线命令与 VIMN 命令各执行一次，读取 `results.pkl`，计算三类指标并保存汇总到 `./result/experiment/intent_vs_baseline.json`。
    - 指标计算位置：`scripts/run_intent_experiment.py:132-155`。
  - 现状：大批次对比汇总文件尚未完成；我已对“VIMN（17 个 ID）”单独做了汇总与扩展分析：
    - `./result/experiment/intent_vimn_only_17.json`
    - `./result/experiment/intent_vimn_only_17_extended.json`

- 现有结果（VIMN，17 个 ID，fast 模式）
  - 指标（简单版）：`cat_match=0.000`，`time_match=0.647`，`json_ok=1.000`
  - 扩展分析：
    - `time_mae_mean=2.59h`（首事件时间绝对误差均值）
    - `cat_jaccard_mean=0.00`（生成与真值的类别集合相似度）
    - `plan_len_diff_mean=-1.59`（生成计划长度偏短）
  - 解读：
    - 时间一致性较好，符合论文强调的 temporal 对齐优势。
    - 类别一致性弱，主要因生成未严格限定历史 POI 格式或类别映射缺失；这与论文指出的语义生成在空间/类别上更具挑战相一致。[1]
    - 全部结果可解析，流程稳定。

- 改进建议（对齐论文方法）
  - Demo 选择增强（保持最小侵入）：
    - 在 `mode=0` 的检索候选中加入“类别一致性过滤”，优先保留与测试日类别分布更匹配的候选作为 `demo`，提升 `cat_match` 与 `cat_jaccard`。
    - 在 prompt 中明确要求使用历史出现的 `POI#ID` 格式，减少不可映射项。
  - 意图输出增强（可选）：
    - 为 VIMN 增加“类别分布头”，进行 next category 预测，利用 Top‑K 类别过滤候选并在 prompt 注入“意图类别摘要”，进一步提升类别一致性。
  - 评估扩展：
    - 时间序列指标：除 `MAE` 可加 `RMSE`、多事件对齐率。
    - 模式一致性：类别序列编辑距离、工作日/周末分布 KL 等。
    - 空间指标（若有坐标）：步长/出行半径分布，呼应论文中的空间—时序度量。[1]

**使用指令速览**
- 单人基线：`python generate.py --dataset 2019 --mode 0 --id <id> --fast`
- 单人 VIMN：`python generate.py --dataset 2019 --mode 0 --id <id> --fast --use_intent --intent_ckpt ./engine/experimental/checkpoints/vimn_lite.pt`
- 批量对比：`python scripts/run_intent_experiment.py --dataset 2019 --ids <逗号ID> --fast --intent_ckpt ./engine/experimental/checkpoints/vimn_lite.pt`
- 汇总查看：`cat ./result/experiment/intent_vimn_only_17.json` 与 `intent_vimn_only_17_extended.json`

**参考**
- [1] Large Language Models as Urban Residents: An LLM Agent Framework for Personal Mobility Generation, arXiv:2402.14744（强调时间维度自一致与语义模式对齐）https://ar5iv.labs.arxiv.org/html/2402.14744
https://arxiv.org/pdf/2411.00823 vimn
https://arxiv.org/pdf/2508.16153 memento

最优解一：CRAG (Corrective RAG) —— 架构优化的最优解来源：ICLR 2024 (Corrective Retrieval Augmented Generation)核心痛点：传统的 Gating 只是决定“用不用”检索。CRAG 提出，如果检索结果质量差，不应该只是退回到直觉（VIMN），而应该采取**“纠正动作”**。算法逻辑：引入一个轻量级的 Retrieval Evaluator（检索评估器），它的输出不仅仅是权重 $\lambda$，而是三个离散动作：Correct (准确)：检索结果完美 $\to$ 直接使用（对应你的 Hybrid Mode）。Ambiguous (模糊)：检索结果似是而非 $\to$ 结合直觉与检索（你的加权融合）。Incorrect (错误)：检索结果完全无关 $\to$ Web Search / Fallback（对应你的 VIMN Only，甚至可以触发反事实推理）。对你的优化（如何改）：把你的 GatingNetwork 升级为 CRAG Evaluator。输入：VIMN 熵 + Memento 分数。输出：分类任务（Correct / Ambiguous / Incorrect）。优势：ICLR 2024 的论文背书，证明这种“纠正式”架构比单纯的“软加权”更鲁棒。
最优解二：Self-RAG (Self-Reflective RAG) —— 生成机制的最优解来源：ICLR 2024 / NeurIPS 2024 (Self-RAG: Learning to Retrieve, Generate, and Critique)核心痛点：RL 训练太难，Reward 太稀疏。算法逻辑：不再训练一个独立的 Gating Network，而是训练模型生成 Reflection Tokens（反思令牌）。模型会自己输出 [Retrieve] 令牌来决定是否调用 Memento。生成后，输出 [IsRel]（是否相关）和 [IsSup]（是否由证据支持）令牌来自我打分。对你的优化（如何改）：这是大改，但效果最好。既然你不想微调 LLM，你可以借鉴其Critic 思想：在训练 Gating Network 时，不仅仅看最终预测对不对（Outcome Reward），还要加入中间评价（Process Reward）。优化点：引入 Process-Supervised Reward Models (PRM) 的概念。
为了毕业论文的创新性与可行性的平衡，我建议采用 "CRAG 的架构思想 + DPO 的训练方法"。

架构上：引用 CRAG (ICLR 2024)，把你的门控网络定义为 "Lightweight Corrective Evaluator"。这比叫 "Gating Net" 更高端。

算法上：引用 DPO (NeurIPS 2023)，放弃不稳定的 REINFORCE，改用 DPO 来训练这个 Evaluator。


**首次实验 2019年 baseline 对比 VIMN 集成（意图检索替换 demo）**
Location Accuracy (Loc-ACC) Evaluation
Agent ID: 934, Dataset: 2019
----------------------------------------
Baseline Loc-ACC: 0.0556 (5/90 correct locations)
VIMN Loc-ACC:     0.0444 (4/90 correct locations)
----------------------------------------
Baseline Acc@1/5/10:
  Acc@1: 0.3846  Acc@5: 0.3846  Acc@10: 0.3846
VIMN Acc@1/5/10:
  Acc@1: 0.7500  Acc@5: 0.7500  Acc@10: 0.7500
  Delta (VIMN - Baseline): -0.0111

- 二次实验：
Agent ID: 934, Dataset: 2019, ExpDir: /home/syf/project/llomob/LLMob/result/ab_test_12_1/exp_2019_934
======================================================================
| Metric       | baseline        | mem             | vimn            |
----------------------------------------------------------------------
| Loc-ACC      | 0.0556 ( 5/90) | 0.0667 ( 6/90) | 0.0444 ( 4/90) |
| Acc@1        | 0.3846          | 0.4286          | 0.7500          |
| Acc@5        | 0.3846          | 0.4286          | 0.7500          |
| Acc@10       | 0.3846          | 0.4286          | 0.7500          |
----------------------------------------------------------------------
| JSD_SD       | 0.0870          | 0.0828          | 0.1012          |
| JSD_SI       | 0.2835          | 0.2821          | 0.2914          |
| JSD_DARD     | 0.4264          | 0.4105          | 0.4973          |
| JSD_STVD     | 0.2600          | 0.2483          | 0.4127          |
----------------------------------------------------------------------
| SD mean (GT) | 1088.3210       |
| SD median (GT) | 286.3013        |
| SI mean (GT) | 123.5526        |
| SI median (GT) | 95.0000         |

- memento训练：

```python scripts/train_memento.py --dataset 2019 --ids all --epochs 20 --top_k 20 --batch_size 8192```

- vimn训练：

```python scripts/train_vimn_global.py --year 2019 --ids 2575 1481 1784 2721 638 7626 1626 7266 1568 2078 2610 1908 2683 1883 3637 225 914 6863 6670 323 3282 2390 2337 4396 7259 1310 3802 1522 1219 1004 4105 540 6157 1556 2266 13 1874 317 2513 3255 934 3599 1775 606 3033 3784 5252 3365 6581 6171 5326 2831 3453 3781 2402 4843 439 1172 3501 1032 2542 1184 1531 6615 7228 1492 6973 67 2680 2956 3138 3638 5765 835 1431 6249 6998 573 884 2356 6463 930 3534 6814 5551 5449 6144 6156 4768 2620 4007 1974 --epochs 2 --batch 256 --amp --device auto ```

- vimn测试：

```python scripts/run_single_ab_test.py --dataset 2019 --id 934 --fast ```


训练：
2575 1481 1784 2721 638 7626 1626 7266 1568 2078 2610 1908 2683 1883 3637 225 914 6863 6670 323 3282 2390 2337 4396 7259 1310 3802 1522 1219 1004 4105 540 6157 1556 2266 13 1874 317 2513 3255 934 3599 1775 606 3033 3784 5252 3365 6581 6171 5326 2831 3453 3781 2402 4843 439 1172 3501 1032 2542 1184 1531 6615 7228 1492 6973 67 2680 2956 3138 3638 5765 835 1431 6249 6998

测试：
573 884 2356 6463 930 3534 6814 5551 5449 6144 6156 4768 2620 4007 1974

# 读取 train_ids.txt 的内容传给 --ids
python -m scripts.train_vimn_global   --year 2019   --ids 2575 1481 1784 2721 638 7626 1626 7266 1568 2078 2610 1908 2683 1883 3637 225 914 6863 6670 323 3282 2390 2337 4396 7259 1310 3802 1522 1219 1004 4105 540 6157 1556 2266 13 1874 317 2513 3255 934 3599 1775 606 3033 3784 5252 3365 6581 6171 5326 2831 3453 3781 2402 4843 439 1172 3501 1032 2542 1184 1531 6615 7228 1492 6973 67 2680 2956 3138 3638 5765 835 1431 6249 6998   --epochs 50   --batch 256   --amp --device auto

python scripts/train_memento.py --dataset 2019 --ids 2575,1481,1784,2721,638,7626,1626,7266,1568,2078,2610,1908,2683,1883,3637,225,914,6863,6670,323,3282,2390,2337,4396,7259,1310,3802,1522,1219,1004,4105,540,6157,1556,2266,13,1874,317,2513,3255,934,3599,1775,606,3033,3784,5252,3365,6581,6171,5326,2831,3453,3781,2402,4843,439,1172,3501,1032,2542,1184,1531,6615,7228,1492,6973,67,2680,2956,3138,3638,5765,835,1431,6249,6998 --epochs 20 --top_k 20 --batch_size 8192

未完成：
python generate.py --dataset 2019 --mode 0 --ids 2575,1481,1784,2721,638,7626,1626,7266,1568,2078,2610,1908,2683,1883,3637,225,914,6863,6670,323,3282,2390,2337,4396,7259,1310,3802,1522,1219,1004,4105,540,6157,1556,2266,13,1874,317,2513,3255,934,3599,1775,606,3033,3784,5252,3365,6581,6171,5326,2831,3453,3781,2402,4843,439,1172,3501,1032,2542,1184,1531,6615,7228,1492,6973,67,2680,2956,3138,3638,5765,835,1431,6249,6998 --resume --use_vimn --vimn_ckpt ./engine/experimental/checkpoints/vimn_global_gru_2019_train_ids.pt

python generate.py --dataset 2019 --mode 0 --ids 2575,1481,1784,2721,638,7626,1626,7266,1568,2078,2610,1908,2683,1883,3637,225,914,6863,6670,323,3282,2390,2337,4396,7259,1310,3802,1522,1219,1004,4105,540,6157,1556,2266,13,1874,317,2513,3255,934,3599,1775,606,3033,3784,5252,3365,6581,6171,5326,2831,3453,3781,2402,4843,439,1172,3501,1032,2542,1184,1531,6615,7228,1492,6973,67,2680,2956,3138,3638,5765,835,1431,6249,6998 --resume --use_memento --vimn_ckpt ./engine/experimental/checkpoints/memento_policy_2019_train_ids.pt

python -m scripts.train_dpo_gating --dataset 2019 --id 934 --generated_base_dir ./result/test_dpo --epochs 3 --lr 1e-3 --beta 0.1 --cost_beta 0.1


python generate.py --dataset 2019 --mode 0 --id 934 --fast --use_vimn --use_memento --use_gating_dpo --vimn_ckpt ./engine/experimental/checkpoints/vimn_global_gru_2019_train_ids.pt --memento_ckpt ./engine/experimental/checkpoints/memento_policy_2019_train_ids.pt --gating_ckpt ./engine/experimental/checkpoints/gating_dpo_934.pt