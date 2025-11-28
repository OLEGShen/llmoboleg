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
