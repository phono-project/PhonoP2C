# PhonoP2C

PhonoP2C（Fast Pinyin-to-Chinese）是一个拼音转汉字的端到端研究项目：输入"中文上下文 + 拼音音节"，输出汉字。新标准架构采用编码器-解码器设计——**pre 模型是因果解码器**（读取中文序列，交叉注意力参考拼音编码器的隐状态，`lm_proj` 输出 `chinese_vocab` logits），**post 模型是双向编码器**（读取拼音序列，输出隐状态与 logits 掩码，不输出 logits）；编码器序列位置 id 位于解码器之后（RoPE）。训练时使用两遍前向（无条件 + 条件），两个 loss（unconditional_loss / conditional_loss）分别记录。

PhonoP2C (Fast Pinyin-to-Chinese) is an end-to-end research project on pinyin-to-Chinese conversion: "Chinese context + pinyin syllables" in, Chinese characters out. The new-standard architecture is encoder-decoder — the **pre model is a causal decoder** (reads the Chinese sequence, cross-attends over the pinyin encoder's hidden states, and outputs `chinese_vocab` logits via `lm_proj`), while the **post model is a bidirectional encoder** (reads the pinyin sequence and outputs hidden states plus a logits mask; no logits). Encoder sequence position ids are placed after the decoder (RoPE). Training uses a two-pass forward (unconditional + conditional) whose losses are logged separately.

## 项目结构

- `main.py` / `tasks/` — Hydra 入口与训练、预处理任务
- `datasets_pipeline/` — 数据层（`dataset.py`、`preprocessor.py`）与共享组件（`constants.py`、`pinyin.py`、`segments.py`）
- `subset.py` / `tokenizer/` — 语料子集抽取与三词表 tokenizer（含 `sample_heteronym` 异读采样）
- `model/` — 解码器 / 编码器、注意力、SwiGLU、MoE、RoPE、KV Cache 自定义算子、`beam_search.py`
- `loss.py` / `metrics/` / `utils/` — 损失、评估指标（含 beam search Top-K 句准确率）与 float8 工具
- `export/` / `demo.py` — Hydra ExecuTorch 导出任务与推理演示
- `config/` / `vocabs/` / `datasets/` / `checkpoints/` — 配置、词表、数据与产出

## Project Layout

- `main.py` / `tasks/` — Hydra entry point and train / preprocess tasks
- `datasets_pipeline/` — data layer (`dataset.py`, `preprocessor.py`) and shared components (`constants.py`, `pinyin.py`, `segments.py`)
- `subset.py` / `tokenizer/` — corpus subsetting and the three-vocabulary tokenizer (incl. `sample_heteronym`)
- `model/` — decoder / encoder, attention, SwiGLU, MoE, RoPE, custom KV-cache ops, `beam_search.py`
- `loss.py` / `metrics/` / `utils/` — losses, metrics (incl. beam-search Top-K sentence accuracy), float8 utilities
- `export/` / `demo.py` — Hydra ExecuTorch export task and inference demo
- `config/` / `vocabs/` / `datasets/` / `checkpoints/` — configs, vocabularies, data, outputs

## 工作流

1. 语料放入 `datasets/pretrain_base`，可选先用 `subset.py` 抽取子集。
2. `python main.py task=preprocess`（或 `python -m datasets_pipeline.preprocessor --preprocess`）将语料规范化为 MDS / Arrow 数据集（`datasets/pretrain_v2`），并统计字-音频率写入 `vocabs/characters_pronounce_frequency.json`。
3. `main.py` 联合训练两个子模型（两遍前向），产出 `checkpoints/`（pre / post 分别保存）。
4. `main.py task=export` 导出 ExecuTorch .pte 文件（pre 多图程序 / post）。
5. `demo.py` 提供参数化的 greedy / beam 推理演示。

## Workflow

1. Put raw corpora in `datasets/pretrain_base`; optionally extract a subset with `subset.py` first.
2. `python main.py task=preprocess` (or `python -m datasets_pipeline.preprocessor --preprocess`) normalizes corpora into MDS / Arrow datasets (`datasets/pretrain_v2`) and writes per-character pronunciation frequencies to `vocabs/characters_pronounce_frequency.json`.
3. `main.py` jointly trains both sub-models (two-pass forward), producing `checkpoints/` (pre / post saved individually).
4. `main.py task=export` exports ExecuTorch `.pte` files (multi-graph pre program / post).
5. `demo.py` provides a parameterized greedy / beam inference demo.

## 运行环境

项目使用 pixi（`pixi.toml`）：Python 3.13、CUDA 13 runtime、cu130 版 PyTorch，以及 jieba、pypinyin、zhconv-rs、streaming（MosaicML）、datasets、hydra-core、wandb、netcal、torchao、bitsandbytes、executorch 和代码质量工具（black、isort、flake8、mypy）。常用命令：

- `pixi install` / `pixi run python ...` 运行任意脚本。
- `python subset.py` — 构建语料子集。
- `python main.py task=preprocess` — 完整预处理（配置见 `config/dataset/pretrain_v2.yaml`）。
- `python main.py` — 训练（可通过 Hydra 覆盖配置）。
- `python -m pytest tests` — 运行测试。
- `python main.py task=export` — ExecuTorch 导出；输入、输出、示例图尺寸、量化与清单配置见 `config/task/export.yaml`。
- `python demo.py --checkpoint <目录> --text <上下文> --pinyin <音节...>` — 推理演示。

> **已知问题的临时说明：** 在当前 Python 3.13 环境下，PyTorch 稳定版对于 NJT 的 `torch.compile` 支持存在已知上游漏洞，表现为 `torch._inductor.exc.InductorError: AssertionError` 符号生成错误。
> 如果您更需要使用 torch.compile 特性，请**按照现在 pixi.toml 默认的状态使用 PyTorch Nightly**，已知 PyTorch Nightly 下可以正常使用 torch.compile 编译该模型。
> 如果您更需要使用稳定版，可在 `config/task/train.yaml` 中将 `compile_model` 设置为 `false`。

## Environment

The project uses pixi (`pixi.toml`): Python 3.13, CUDA 13 runtime, PyTorch cu130 build, plus jieba, pypinyin, zhconv-rs, streaming (MosaicML), datasets, hydra-core, wandb, netcal, torchao, bitsandbytes, executorch, and quality tools (black, isort, flake8, mypy). Typical workflow commands:

- `pixi install` / `pixi run python ...` to run any script.
- `python subset.py` — build a corpus subset.
- `python main.py task=preprocess` — full preprocessing (config in `config/dataset/pretrain_v2.yaml`).
- `python main.py` — train (configuration can be overridden via Hydra).
- `python -m pytest tests` — run the test suite.
- `python main.py task=export` — ExecuTorch export; inputs, outputs, example graph dimensions, quantization, and manifest options are configured in `config/task/export.yaml`.
- `python demo.py --checkpoint <dir> --text <context> --pinyin <syllables...>` — inference demo.

> **Temporary note on known issues:** In the current Python 3.13 environment, the stable version of PyTorch has a known upstream bug regarding support for NJT’s `torch.compile`, which manifests as a symbol generation error `torch._inductor.exc.InductorError: AssertionError`.
> If you require the `torch.compile` feature, please **use PyTorch Nightly as is currently the default in pixi.toml**; it is known that `torch.compile` works correctly when compiling this model with PyTorch Nightly.
> If you prefer to use the stable release, you can set `compile_model` to `false` in `config/task/train.yaml`.

## 文档

项目文档按预处理、训练、导出与推理三个部门组织，每份文档按部门层和函数层描述组件，详细用法请阅读对应文档：

- [预处理部门（preprocess.md）](docs/zh-cn/preprocess.md) — `subset.py`、`datasets_pipeline/preprocessor.py`、`tokenizer/`、数据层
- [训练部门（train.md）](docs/zh-cn/train.md) — 模型架构、`main.py`、hydra 配置、Trainer、损失与指标
- [导出与推理部门（export.md）](docs/zh-cn/export.md) — ExecuTorch 导出与推理演示

详细中文文档见 [docs/zh-cn/](docs/zh-cn/README.md)。

## Documentation

The project documentation is organized into three departments — preprocess, train, export & inference — each describing components at department and function levels. For details, see:

- [Preprocess department (preprocess.md)](docs/en-us/preprocess.md) — `subset.py`, `datasets_pipeline/preprocessor.py`, `tokenizer/`, data layer
- [Train department (train.md)](docs/en-us/train.md) — model architecture, `main.py`, hydra configs, Trainer, losses and metrics
- [Export & Inference department (export.md)](docs/en-us/export.md) — ExecuTorch export and inference demo

Detailed documentations can be found at [docs/en-us/](docs/en-us/README.md).

## 许可证与最终声明

由于本人精力有限，而且文档大部分使用了 LLM 生成，不可避免的会存在纰漏、更新不及时等问题。如果发现有任何问题，欢迎提出 issues。

本项目基于 Apache License 2.0 开源，详见 [LICENSE](LICENSE)。

## License and Final Note

Due to my limited time and resources, and since most of this documentation was generated by an LLM, there may inevitably be errors or outdated information. If you find any issues, please feel free to open an issue.

Open-sourced under the Apache License 2.0; see [LICENSE](LICENSE).
