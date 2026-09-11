# 导出与推理部门（zh-CN）

## 1. 部门概述

导出与推理部门覆盖训练完成后、使用冻结模型权重的各阶段：

- **解码校准** — `tasks/param_search.py` 在验证集子集上运行训练好的模型， 抽取每个位置的候选概率，构建词典 Trie，并用 Optuna 搜索最优的 Viterbi 解码先验（`beta_single`、`beta_word`）。算法位于 `algo/`，校准词典位于 `dicts/`。
- **设备端导出** — `export.py` 冻结并导出两个子模型为 ExecuTorch `.pte` 文件，采用 XNNPACK 动态逐通道量化，并保留共享交叉注意力 KV Cache 语义。
- **推理演示** — `demo.py` 通过参数化 CLI 在 PyTorch 中运行完整的 greedy / beam-search 流水线。

组件清单：

| 组件 | 文件 | 职责 |
|---|---|---|
| 解码校准 | `tasks/param_search.py` | 词典+Trie 构建、概率抽取、Optuna 搜索 Viterbi 先验 |
| Trie 匹配 | `algo/trie.py` | 词典加载/构建/保存、逐位置词匹配 |
| Viterbi DP | `algo/viterbi_dp.py` | 带先验的 N-best 束搜索解码 |
| 词典资产 | `dicts/dict_v1.txt` | 校准词典（多字词） |
| ExecuTorch 导出 | `export.py` | torch.export + XNNPACK 量化 -> `.pte` 文件 |
| 推理演示 | `demo.py` | 参数化 KV Cache 推理：greedy 与 beam search |

## 2. `tasks/param_search.py` — 解码超参数搜索

**功能：** 在预训练（冻结）模型上校准词典约束的 Viterbi 解码。两个子任务：

1. **预处理**：过滤词典（chinese 词表检查、仅多字词、长度 < max、 去重），构建并保存 Trie，在验证集子集上运行模型推理，抽取每个位置的非 零候选概率并降序排列，把 token id 转换为汉字，保存为 `probs.json`。
2. **Optuna 搜索**：加载 `probs.json` 与 `dict_trie.json`，并行预计算每个 样本位置的 Trie 词匹配，以分数 `sum(ln probs) + n_words·ln(beta)` 评估 Viterbi N-best 解码，优化 `beta_single` / `beta_word`，保存 `results.json`。

**用法：** `python main.py task=param_search`（配置 `config/task/param_search.yaml`；关键字段：`pretrained_pre_model`、 `pretrained_post_model`、`vocabs_config`、`dict_path`、`val_dir`、 `output_dir`、`num_samples`、`batchsize`、`epsilon`、`subtasks.*`、 `dict.max_len: 7`、`optuna.n_trials: 100`、`optuna.n_best: 5`、 `optuna.objective: "sentence_acc"`、beta 范围）。

### 模块状态与 worker

#### `init_worker_trie(trie)`
- 功能：进程池初始化函数，把共享 Trie 安装到每个 worker 的 `_global_trie`。
- 行为：写入全局变量；后续 worker 调用可直接使用。

#### `process_json_lines_chunk(lines)`
- 功能：在 worker 中解析一段 `probs.json` 行并预计算 Trie 词匹配。
- 行为：Trie 未初始化时抛出 `ValueError`；对每行构建逐位置 `{char: prob}` 候选字典，对每个位置运行 `find_matching_words`，返回 `(candidates, words_at, target)` 元组。

#### `evaluate_chunk_viterbi(chunk, beta_single, beta_word, N)`
- 功能：在 worker 进程中对预计算的数据块评估 Viterbi 解码。
- 行为：对每个样本运行 `viterbi_nbest`；统计目标作为最优（rank-1）结果与 出现在 top-N 内的样本数；返回 `(correct_1, correct_N, total)`。

### `ParamSearchRunner`

#### `__init__(cfg)`
- 功能：准备运行器。
- 行为：读取设备与任务配置；创建输出目录；构建分词器。

#### `run()`
- 功能：运行已启用的子任务。
- 行为：`subtasks.preprocess`（默认 true）时运行 `_subtask_preprocess`，然后 `subtasks.optuna_search`（默认 true）时运行 `_subtask_optuna_search`。

#### `_subtask_preprocess()`
- 功能：词典处理 + 推理概率抽取。
- 行为：加载词典（`load_dictionary`），过滤掉不是全部在 chinese 词表内的 词、单字词、长度 ≥ `dict.max_len` 的词，按出现顺序去重；构建并保存 `dict_trie.json`；加载预训练模型（并把可能性掩码安装到 post 模型）；加载 验证集，打乱后取前 `num_samples` 条，应用验证变换；在 bf16 autocast 下 推理（pre -> post，NJT 路径）；对每个样本把 logits 转概率，保留大于 `epsilon` 的候选，降序排序，经分词器把 id 映射为汉字，向 `probs.json` 追加一行 JSON `{"target", "positions"}`；每 10 个批次打印进度。

#### `_subtask_optuna_search()`
- 功能：Optuna 搜索解码先验。
- 行为：加载 Trie；读取全部 `probs.json` 行；在 `os.cpu_count()` 个 worker 上分块并预计算词匹配（块大小 ≈ 行数 / (workers · 4)）；把预计算数据切成 每 worker 一块的评估块；创建 Optuna study（最大化、 `GPSampler(seed=42)`）；按所选目标函数运行 `n_trials` 次；把最优 `beta_single`、`beta_word`、sentence-ACC、N-sentence-ACC、`n_best` 与 `n_trials` 写入 `results.json`；打印汇总。

#### `_objective_s_acc(trial)`
- 功能：优化 rank-1 句子准确率的 Optuna 目标函数。
- 行为：在 [`beta_single_low`, `beta_single_high`]（线性）中建议 `beta_single`，在 [`beta_word_low`, `beta_word_high`]（线性）中建议 `beta_word`；并行评估所有块；计算 sentence-ACC（目标等于最优结果）与 N-sentence-ACC（目标出现在 top-N 内）；两者都存入 trial user attrs； 返回 sentence-ACC。

#### `_objective_n_s_acc(trial)`
- 功能：优化 top-N 句子准确率的 Optuna 目标函数。
- 行为：与上相同，但 beta 用 `log=True` 采样，返回 N-sentence-ACC。

## 3. `algo/trie.py` — Trie 词典匹配

**功能：** 从中文词表构建/加载嵌套字典 Trie，并在给定逐位置字符概率时查找 可以从某位置起始的全部词典词。

### 函数

#### `load_dictionary(path)`
- 功能：读取按行分隔的词表。
- 行为：去除首尾空白；跳过空行。

#### `build_trie(words)`
- 功能：构建嵌套字典 Trie。
- 行为：每个词逐字用 `setdefault` 插入；`"#"` 键标记词尾；返回 Trie 根 字典。

#### `load_trie(path)` / `save_trie(trie, path)`
- 功能：Trie 的 JSON（反）序列化。
- 行为：`ensure_ascii=False`，汉字保持可读。

#### `find_matching_words(trie, candidates, start)`
- 功能：查找可以从 `start` 位置起始的全部词典词。
- 用法：`param_search.py` 对每个样本预计算一次。
- 行为：对 Trie 节点做迭代 DFS；词只能经由在当前候选字典中存在且概率 > 0 的字符延伸；每个完成的词记录为 `(length, word_str, log_prob_sum)`；结果 按长度降序排列。

## 4. `algo/viterbi_dp.py` — Viterbi N-Best 解码

**功能：** 词典约束的 P2C 解码 Viterbi DP 束搜索。路径分数为 `sum(ln probs)` 加每次转移的先验惩罚：单字步 `ln(beta_single)`，多字词典 词 `ln(beta_word)`。

### `viterbi_nbest(candidates, words_at, beta_single, beta_word, N)`
- 功能：返回候选序列的 top-N 切分。
- 用法：由参数搜索中的 `evaluate_chunk_viterbi` 调用。
- 行为：把候选概率转换到对数空间（非正值记 `-inf`）；维护 `paths[pos]`，以已解码前缀字符串为键的字典（合并相同前缀），值为 `(score, 回溯节点)`；每个位置把束裁剪到 top-N 个唯一前缀；同时推进单字 与多字 Trie 词；结束时对最终状态排序，保留 top-N，经回溯节点还原词表。 返回按分数降序的 `[(score, words), ...]`；空候选列表返回 `[(0.0, [])]`；终点不可达时返回 `[]`。

## 5. `dicts/` — 校准词典

`dicts/dict_v1.txt` — 从 jieba 默认词典派生的、按行分隔的大规模中文词表（约 34.9 万词条），用作解码词典。词必须通过 chinese 词表检查、为多字词、且长度小于 `dict.max_len`（7），才会进入 Trie。由 `tasks/param_search.py` 消费；归属与许可详情见 `THIRD_PARTY_NOTICES.md`。

## 6. `export.py` — ExecuTorch 导出

**功能：** 把训练好的 `PhonoP2CPreModel` 与 `PhonoP2CPostModel` 导出为带 XNNPACK 动态逐通道量化的 ExecuTorch `.pte` 程序。两个 pre pass 作为命名方法写入同一个多方法 `pre_model.pte`；post hidden states 与 logits mask 在运行时传给 pre 条件方法。

**用法：** `python export.py`。配置常量：`CHECKPOINT_DIR` （`./checkpoints/v2_0-base-alpha05/final_model`）、`MODEL_TYPE` （`torch.float32`）、`SAVE_DIR`（`./export_output`）。

### `load_model_from_checkpoint(checkpoint_dir, device)`
- 功能：从 checkpoint 目录加载两个子模型。
- 行为：要求目录包含 `pre_model/` 与 `post_model/` 两个子目录（由 `save_pretrained` 产生，config.json + safetensors）；经 `from_pretrained` 加载、移到指定设备，两个模型都切换为 eval 模式；返回 `(pre_model, post_model)`。

### 顶层脚本行为
- 加载模型，转成 `MODEL_TYPE`，冻结全部参数（`requires_grad = False`）。
- 从配置推导 self-attention 缓存几何 `(mhsa_layers, 2, B, pre_max, pre_nheads, pre_head_dim)`。
- 为 causal pre pass、conditional pre pass 与 post encoder 构建 dummy 输入。条件 pass 接收 post hidden states 和一行 logits mask；不分配 cross-KV Cache。
- 声明动态维度：`new_prefix_len`（1..pre_max）用于 pre 输入 id， `post_len`（1..post_cfg.pre_max_seqlen）用于 post 输入 id。
- 用 `torch.export.export`（动态形状）导出两个模型并打印计算图。
- 经 `XNNPACKQuantizer` + `get_symmetric_quantization_config( is_per_channel=True, is_dynamic=True)`，用 torchao 的 `prepare_pt2e` / `convert_pt2e` 量化，中间在 `no_grad` 下跑一次 dummy forward 做校准； 再导出量化后的模型。
- 将两个 pre 图量化后分别用 `XnnpackPartitioner(per_op_mode=True)` lower，再组合 edge 方法并合并写出 `pre_model.pte`；这样可避开 ExecuTorch 1.4.1 的 dependency-cycle 问题，同时共享常量；post 图单独写出为 `post_model.pte`。
- 用 `MemoryPlanningPass(alloc_graph_input=False)` 构建 ExecuTorch 程序。

## 7. `demo.py` — 推理演示

**功能：** 在 PyTorch 中用 self-attention KV Cache 运行训练好的 greedy 与 beam-search 流水线。

**用法：**

```bash
python demo.py \
  --checkpoint checkpoints/<run>/final_model \
  --text "上下文" \
  --pinyin pin yin \
  --beam-size 3 \
  --device auto \
  --dtype auto
```

checkpoint 与拼音参数必填。设备和精度默认在 CUDA 可用时选择 CUDA/BF16，
否则选择 CPU/FP32；模块函数仍可导入用于交互调用。

### `load_model_from_checkpoint(checkpoint_dir, device)`
- 功能：与 `export.py` 相同的加载器。
- 行为：从 checkpoint 目录加载 `pre_model/` 与 `post_model/`，eval 模式。

### `create_pre_kv_cache(pre_model, batch_size=1, device=device, dtype=torch.float32)`
- 功能：分配推理用 self-attention KV Cache。
- 行为：分配 `(pre_num_layers, 2, B, pre_max, pre_nheads, pre_head_dim)`，不分配 cross-KV；`beam_search` 负责一次调用中的 cache 生命周期。

### `predict_step(text, pinyin_list, pre_model, post_model, tokenizer, device, topk=1)`
- 功能：面向 greedy 或 top-k beam generation 的公开 PyTorch 演示包装器。
- 行为：编码上下文与拼音、调用 `beam_search`，返回解码文本或 N-best beams。

### `build_parser()` / `resolve_runtime(device_name, dtype_name)`
- 功能：定义并校验 CLI 参数，脚本不再嵌入本地 checkpoint 或设备。
- 行为：`auto` 在 CUDA 可用时解析为 CUDA/BF16，否则为 CPU/FP32。

### `main(argv=None)`
- 行为：加载指定 checkpoint 与 tokenizer，运行 greedy 和可选的 N-best beam search；计时时同步 CUDA，并打印解码候选。
