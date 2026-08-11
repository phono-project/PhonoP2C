# PhonoP2C

PhonoP2C（Fast Pinyin-to-Chinese）是一个拼音转汉字的端到端研究项目：输入"中文上下文 + 拼音音节"，输出汉字。模型采用两段式 PostfixLM 架构，由前段因果编码器处理上下文、后段双向解码器读取拼音并预测汉字，候选汉字以拼音可能性掩码限定，并可用词典约束的 Viterbi 解码精化；两个子模型联合端到端训练。

PhonoP2C (Fast Pinyin-to-Chinese) is an end-to-end research project on pinyin-to-Chinese conversion: "Chinese context + pinyin syllables" in, Chinese characters out. It uses a two-stage PostfixLM architecture — a causal pre-model encodes the context, a bidirectional post-model reads pinyin and predicts characters, restricted by a pinyin possibility mask and optionally refined with dictionary-constrained Viterbi decoding. Both sub-models are trained jointly end-to-end.

## 项目结构

- `main.py` / `tasks/` — hydra 入口与训练、解码校准任务
- `preprocessor.py` / `subset.py` / `tokenizer.py` / `dataset.py` — 语料预处理、分词与数据层
- `model/` — PreModel / PostModel、注意力、SwiGLU、MoE、RoPE、KV Cache 自定义算子
- `loss.py` / `metrics/` / `utils/` — 损失、评估指标与 float8 工具
- `algo/` — Trie 词典匹配与 Viterbi N-best 解码
- `export.py` / `demo.py` — ExecuTorch 导出与推理演示
- `config/` / `vocabs/` / `dicts/` / `datasets/` / `checkpoints/` — 配置、词表、词典、数据与产出

## Project Layout

- `main.py` / `tasks/` — hydra entry point and train / decoding-calibration tasks
- `preprocessor.py` / `subset.py` / `tokenizer.py` / `dataset.py` — corpus preprocessing, tokenization, data layer
- `model/` — PreModel / PostModel, attention, SwiGLU, MoE, RoPE, custom KV-cache ops
- `loss.py` / `metrics/` / `utils/` — losses, evaluation metrics, float8 utilities
- `algo/` — trie dictionary matching and Viterbi N-best decoding
- `export.py` / `demo.py` — ExecuTorch export and inference demo
- `config/` / `vocabs/` / `dicts/` / `datasets/` / `checkpoints/` — configs, vocabularies, dictionaries, data, outputs

## 工作流

1. 语料放入 `datasets/pretrain_base`，可选先用 `subset.py` 抽取子集。
2. `preprocessor.py` 将语料规范化为 MDS / Arrow 数据集（`datasets/pretrain_v1`）。
3. `main.py` 联合训练两个子模型，产出 `checkpoints/`。
4. `main.py task=param_search` 用 Optuna 搜索 Viterbi 解码先验。
5. `export.py` 导出 ExecuTorch .pte 文件。
6. `demo.py` 演示 greedy / top-k / Viterbi 解码。

## Workflow

1. Put raw corpora in `datasets/pretrain_base`; optionally extract a subset with `subset.py` first.
2. `preprocessor.py` normalizes corpora into MDS / Arrow datasets (`datasets/pretrain_v1`).
3. `main.py` jointly trains both sub-models, producing `checkpoints/`.
4. `main.py task=param_search` tunes Viterbi decoding priors with Optuna.
5. `export.py` exports ExecuTorch `.pte` files.
6. `demo.py` demonstrates greedy / top-k / Viterbi decoding.

## 运行环境

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

## Environment

The project uses pixi (`pixi.toml`): Python 3.13, CUDA 13 runtime, PyTorch cu130 build, plus jieba, pypinyin, zhconv-rs, streaming (MosaicML), datasets, hydra-core, wandb, optuna, netcal, torchao, bitsandbytes, executorch, and quality tools (black, isort, flake8, mypy). Typical workflow commands:

- `pixi install` / `pixi run python ...` to run any script.
- `python subset.py` — build a corpus subset.
- `python preprocessor.py --preprocess` — full preprocessing.
- `python preprocessor.py --generate_val` — materialize the val dataset.
- `python main.py` — train (config overridable via hydra, e.g. `task=param_search`).
- `python export.py` — ExecuTorch export.
- `python demo.py` — inference demo.

> **Temporary note on known issues:** In the current Python 3.13 environment, the stable version of PyTorch has a known upstream bug regarding support for NJT’s `torch.compile`, which manifests as a symbol generation error `torch._inductor.exc.InductorError: AssertionError`.
> If you require the `torch.compile` feature, please **use PyTorch Nightly as is currently the default in pixi.toml**; it is known that `torch.compile` works correctly when compiling this model with PyTorch Nightly.
> If you prefer to use the stable release, you can set `compile_model` to `false` in `config/task/train.yaml`.

## 文档

项目文档按预处理、训练、导出与推理三个部门组织，每份文档按部门层和函数层描述组件，详细用法请阅读对应文档：

- [预处理部门（preprocess.md）](docs/zh-cn/preprocess.md) — `subset.py`、`preprocessor.py`、`tokenizer.py`、数据层
- [训练部门（train.md）](docs/zh-cn/train.md) — 模型架构、`main.py`、hydra 配置、Trainer、损失与指标
- [导出与推理部门（export.md）](docs/zh-cn/export.md) — 解码校准、ExecuTorch 导出、推理演示

详细中文文档见 [docs/zh-cn/](docs/zh-cn/README.md)。

## Documentation

The project documentation is organized into three departments — preprocess, train, export & inference — each describing components at department and function levels. For details, see:

- [Preprocess department (preprocess.md)](docs/en-us/preprocess.md) — `subset.py`, `preprocessor.py`, `tokenizer.py`, data layer
- [Train department (train.md)](docs/en-us/train.md) — model architecture, `main.py`, hydra configs, Trainer, losses and metrics
- [Export & Inference department (export.md)](docs/en-us/export.md) — decoding calibration, ExecuTorch export, inference demo

Detailed documentations can be found at [docs/en-us/](docs/en-us/README.md).

## 许可证与最终声明

由于本人精力有限，而且文档大部分使用了 LLM 生成，不可避免的会存在纰漏、更新不及时等问题。如果发现有任何问题，欢迎提出 issues。

本项目基于 Apache License 2.0 开源，详见 [LICENSE](LICENSE)。

## License and Final Note

Due to my limited time and resources, and since most of this documentation was generated by an LLM, there may inevitably be errors or outdated information. If you find any issues, please feel free to open an issue.

Open-sourced under the Apache License 2.0; see [LICENSE](LICENSE).
