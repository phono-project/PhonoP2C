# Export & Inference Department (en-US)

## 1. Department Overview

This department covers two post-training operations:

- `export/` implements the Hydra task that exports the pre and post models to ExecuTorch `.pte` programs.
- `demo.py` runs greedy or beam-search inference from a PyTorch checkpoint.

## 2. `export/` — ExecuTorch Export Task

The exporter loads `pre_model/` and `post_model/` from a checkpoint directory. It produces a multi-method `pre_model.pte`, a `post_model.pte`, and selective build manifests containing the operators and dtype variants required by both programs.

The pre program contains separate methods for prefix decoding, conditional decoding, and cross-attention KV projection. Export uses dynamic sequence dimensions and optionally applies XNNPACK dynamic per-channel quantization.

Run the task through the common entry point:

```bash
python main.py task=export
```

`config/task/export.yaml` defines the input checkpoint, output directory and filenames, target device and dtype, model metadata, representative input dimensions, quantization mode (`none`, `w8a8`, or `w4a8`), graph strictness and printing, memory-planning behavior, and selective-build manifest settings. Every field can be overridden through Hydra, for example:

```bash
python main.py task=export \
  task.checkpoint_dir=checkpoints/my-run/final_model \
  task.output_dir=export_output/my-run \
  task.quantization.mode=w8a8
```

`export/task.py` validates the runtime configuration, resolves paths against Hydra's original working directory, captures the four dynamic graphs, applies quantization, lowers them with XNNPACK, writes both programs, and optionally generates per-model and merged operator manifests.

## 3. `demo.py` — Inference Demo

The demo accepts the checkpoint, vocabulary configuration, context text, pinyin syllables, beam size, device, and dtype as command-line options:

```bash
python demo.py \
  --checkpoint checkpoints/<run>/final_model \
  --text "" \
  --pinyin ni hao \
  --beam-size 3
```

It first runs greedy decoding and then, when the beam size is greater than one, prints N-best beam-search results and execution time.

## 4. `tools/build_pack_v2_2.py` — phono-core Package Builder

The versioned package builder validates exported programs, Hugging Face model configurations, and tokenizer vocabularies before assembling a phono-core model package. Its filename and `MODEL_FORMAT_VERSION` are fixed at v2.2; a future package schema should use a new builder instead of changing this file.

Required inputs are the pre/post PTE programs, their respective `PhonoP2CPreModel` and `PhonoP2CPostModel` `config.json` files, the Chinese/context/pinyin vocabularies, an output directory, and a model version beginning with `v2_2-`. A segment PTE and its character vocabulary are optional and must be provided together; its Hugging Face config may also be supplied for additional checks.

```bash
python tools/build_pack_v2_2.py \
  --pre-model export_output/pre_model.pte \
  --post-model export_output/post_model.pte \
  --pre-config checkpoints/<run>/final_model/pre_model/config.json \
  --post-config checkpoints/<run>/final_model/post_model/config.json \
  --chinese-vocab vocabs/chinese_vocab.txt \
  --context-vocab vocabs/context_vocab.txt \
  --pinyin-vocab vocabs/pinyin_vocab.txt \
  --model-version v2_2-<name> \
  --output-dir <package-dir>
```

The builder parses every PTE with the installed ExecuTorch runtime and checks exact method names, input/output shapes, batch width, sequence limits, KV-cache dimensions, cross-KV compatibility, and candidate width. It also validates model architecture identifiers, attention dimensions, vocabulary sizes, duplicate tokens, and special-token collisions. With a segment model, it additionally verifies the `Tensor[1,L] -> Tensor[1,L-1]` contract and the configured input range. Output is staged in a temporary sibling directory and renamed only after every copy succeeds; an existing output directory is never overwritten.

The resulting layout is `config.json`, `bins/pre_model.pte`, `bins/post_model.pte`, the three files under `vocabs/`, and optionally `bins/pinyin_segment.pte` plus `vocabs/pinyin_char_vocab.txt`. `config.json` binds the package to string format version `"2.2"` and the method/path names expected by phono-core.
