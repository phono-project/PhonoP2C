# PhonoP2C — 项目文档（zh-CN）

## 1. PhonoP2C 是什么

PhonoP2C（Fast Pinyin-to-Chinese）是 PhonoP2C-collection 下的一个研究子项目，实现了一个基于两段式 "PostfixLM" 架构的**拼音转汉字（P2C）模型**：

- **PhonoP2CPreModel** — 带两个导出 pass 的因果解码器：无条件 pass 填充 self-attention 历史，条件 pass 对 post hidden states 做交叉注意力并预测汉字。
- **PhonoP2CPostModel** — 双向拼音编码器，返回 hidden states 与**拼音->汉字可能性掩码**；不维护 cross-KV，也不直接输出汉字 logits。

因此输入是"中文上下文 + 拼音音节"，输出是"汉字"。两个子模型在 `(上下文前缀, 拼音, 汉字目标)` 样本上端到端联合训练。

## 2. 仓库结构

| 路径 | 用途 |
|---|---|
| `main.py` | hydra 入口；分发 train / param_search 两类任务 |
| `preprocessor.py` | 预处理流水线：原始语料转 MDS / Arrow 数据集 |
| `subset.py` | 从语料中抽取随机子集并输出 parquet |
| `tokenizer.py` | 三词表分词器（chinese / context / pinyin） |
| `dataset.py` | 训练期数据变换、NJT collate、流式数据集 |
| `loss.py` | 损失函数（交叉熵、focal、掩码感知的 label smoothing） |
| `export.py` | 将前/后段模型导出为 ExecuTorch .pte 文件 |
| `demo.py` | 推理演示（greedy / top-k / Viterbi） |
| `tasks/train.py` | Trainer：联合训练循环、验证、checkpoint |
| `tasks/param_search.py` | Optuna 搜索 Viterbi 解码先验（beta_single/beta_word） |
| `model/config.py` | PreModelConfig / PostModelConfig 及 YAML 转配置的构建器 |
| `model/model.py` | PhonoP2CPreModel / PhonoP2CPostModel |
| `model/attn.py` | MHSALayer（自注意力）/ MHCALayer（交叉注意力） |
| `model/ffn.py` | SwiGLU 前馈层 |
| `model/moe.py` | MoE_EC_FFN（expert-choice MoE 前馈，可选） |
| `model/utils.py` | RoPE（RotaryEmbedding）辅助函数、局部位置 id |
| `model/custom_ops.py` | 自定义 torch.library KV Cache 更新算子（导出友好） |
| `algo/trie.py` | Trie 构建 / 加载 / 词典词匹配 |
| `algo/viterbi_dp.py` | Viterbi DP N-best 束搜索解码 |
| `metrics/accumulator.py` | ACC / Top-k ACC / Sentence-ACC / Adaptive ECE 累加器 |
| `utils/float8.py` | torchao float8 训练转换的模块过滤函数 |
| `config/` | hydra 配置（model、dataset、task、system、logging、output） |
| `vocabs/` | chinese / context / pinyin 词表及 config.yaml |
| `dicts/` | 解码用校准词典（dict_v1.txt 来源于 jieba） |
| `datasets/` | 原始语料（位于 pretrain_base）与生成的数据集（pretrain_v2） |
| `checkpoints/` | 训练输出（pre_model / post_model 子目录） |
| `pixi.toml` | pixi 环境定义（Python 3.13、CUDA 13、torch cu130） |
| `docs/zh-cn/` | 本文档集（中文） |
| `docs/en-us/` | 本文档集的英文版 |

## 3. 三个部门

文档按三个部门（项目生命周期的三个阶段）组织，每个部门对应一份文档：

| 部门 | 文档 | 覆盖内容 | 入口命令 |
|---|---|---|---|
| 预处理 | `preprocess.md` | 原始语料转模型样本：`subset.py`、`preprocessor.py`、`tokenizer.py`、`vocabs/`，以及 `dataset.py` 的数据层 | `python subset.py`、`python preprocessor.py --preprocess` |
| 训练 | `train.md` | 模型架构与联合训练流水线：`main.py`、`config/`、`tasks/train.py`、`model/`、`loss.py`、`metrics/`、`utils/float8.py` | `python main.py` |
| 导出与推理 | `export.md` | 训练后的各阶段：解码校准（`tasks/param_search.py`、`algo/`、`dicts/`）、设备端导出（`export.py`）、推理演示（`demo.py`） | `python main.py task=param_search ...`、`python export.py`、`python demo.py` |

### 3.1 预处理部门

原始语料（JSONL、纯文本、parquet）经过规范化（简繁转换、NFKC、去 emoji占位符、context 词表过滤），切分为中文 / 停顿标点 / 非中文三类连续片段，再切成长度受限的样本，并为每个中文字符标注嵌套的拼音读音。训练集以 MDS（MosaicML StreamingDataset）格式并行写出，验证集以 HF Arrow 格式写出并物化 prefix/suffix/pinyin 字段。`subset.py` 是从大语料中抽取随机子集的辅助工具（例如构建较小的 fineweb 子集）。

### 3.2 训练部门

hydra 驱动的联合训练器：根据配置构建两个子模型，由分词器计算拼音->汉字可能性掩码，在 NJT（嵌套锯齿张量）批次上用 BF16 混合精度端到端联合训练两个模型，支持可选的 torchao float8 线性层加速、`torch.compile`、AdamW/AdEMAMix 系优化器、余弦学习率调度、梯度裁剪、周期性验证（loss、ACC、Top-k ACC、Sentence-ACC、Adaptive ECE）、wandb 日志，以及 `save_pretrained` checkpoint 保存。

### 3.3 导出与推理部门

训练完成后：`param_search.py` 用冻结模型跑验证集子集，抽取每个位置的候选概率，构建词典 Trie，并用 Optuna 搜索 Viterbi 解码先验`beta_single` / `beta_word`；`export.py` 冻结并导出包含两个 pre 方法的多方法程序和 post 编码器为 ExecuTorch`.pte` 文件，采用 XNNPACK 动态逐通道量化；`demo.py` 在 PyTorch 中运行同样的流水线（greedy、top-k、以及词典约束的 Viterbi 解码）。

## 4. 端到端数据流（文字描述）

1. **语料获取** — 原始数据建议存放于 `datasets/pretrain_base`，数据处理器支持 LCCC、MMC、CLUE、wikipedia、zhihu-kol、fineweb 等 JSONL、Parquet 格式的数据。超大语料可先用 `subset.py` 抽取子集。
2. **预处理** — `preprocessor.py` 规范化每条文本，分段，切成 16–64 字符的样本，计算嵌套的逐字拼音，输出 `datasets/pretrain_v2/train`（MDS，默认zstd压缩）和 `datasets/pretrain_v2/val`（HF Arrow，物化 prefix/suffix/pinyin 对）。
3. **训练** — `main.py` 加载 hydra 配置；`Trainer` 构建分词器、可能性掩码、两个模型，并流式读取 MDS 训练批次。每个批次在线变换（片段选择、拼音增强）后 collate 为 NJT，送入 pre -> post 模型。post 模型的 logits 被可能性掩码过滤后计算损失，两个模型联合优化。验证集为 Arrow 格式，周期性验证。每轮保存 `pre_model` / `post_model` checkpoint。
4. **解码校准** — `main.py task=param_search` 在验证集子集上推理，保存每个位置的概率候选与词典 Trie，再由 Optuna 搜索 Viterbi N-best 解码的最优 `beta_single` / `beta_word`。
5. **导出** — `export.py` 加载最终 checkpoint，导出两个 pre 方法和 post 编码器，应用 XNNPACK 动态量化，写出 `pre_model.pte` / `post_model.pte`。
6. **推理** — `demo.py` 编码拼音一次，再使用带 self-KV Cache 的 pre 条件 pass，用 greedy、top-k 或词典约束的 Viterbi 解码完成拼音转汉字。

## 5. 运行环境

项目使用 pixi（`pixi.toml`）：Python 3.13、CUDA 13 runtime、cu130 版 PyTorch，以及 jieba、pypinyin、zhconv-rs、streaming（MosaicML）、datasets、hydra-core、wandb、optuna、netcal、torchao、bitsandbytes、executorch 和代码质量工具（black、isort、flake8、mypy）。常用命令：

- `pixi install` / `pixi run python ...` 运行任意脚本。
- `python subset.py` — 构建语料子集。
- `python preprocessor.py --preprocess` — 完整预处理。
- `python preprocessor.py --generate_val` — 物化验证集。
- `python main.py` — 训练（可通过 hydra 覆盖配置，例如 `task=param_search`）。
- `python export.py` — ExecuTorch 导出。
- `python demo.py` — 推理演示。

> **已知问题的临时说明：** 在当前 Python 3.13 环境下，PyTorch 稳定版对于 NJT 的 `torch.compile` 支持存在已知上游漏洞，表现为 `torch._inductor.exc.InductorError: AssertionError` 符号生成错误。
> 如果您更需要使用 torch.compile 特性，请**按照现在 pixi.toml 默认的状态使用 PyTorch Nightly**，已知 PyTorch Nightly 下可以正常使用 torch.compile 编译该模型。
> 如果您更需要使用稳定版，可在 `config/task/train.yaml` 中将 `compile_model` 设置为 `false`。

## 6. 阅读指南

每份部门文档按两层抽象描述组件：

- **部门层** — 部门的用途、输入输出、整体处理流程。
- **函数层** — 每个模块、类、函数：功能（做什么）、用法（如何调用及参数）、 行为（运行时发生什么，包括边界情况）。
