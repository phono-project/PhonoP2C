# 预处理部门（zh-CN）

## 1. 部门概述

预处理部门把原始文本语料转换为训练部门可直接消费的样本。输入是原始语料 文件（JSONL、纯文本、parquet）和词表定义文件；输出是：

- `datasets/pretrain_v1/train` — MDS（MosaicML StreamingDataset）训练集， 分片并压缩（zstd），每个样本包含 `text_list`、`pinyin_list`（嵌套的逐字 读音，JSON）和 `labels`。
- `datasets/pretrain_v1/val` — HF Arrow 验证集，每个样本物化了 `prefix` / `suffix` / `pinyin` 字段。
- 中间产物 `datasets/pretrain_v1/train_original` 和 `datasets/pretrain_v1/val_original`。

本部门包含：

| 组件 | 文件 | 职责 |
|---|---|---|
| 语料子集抽取 | `subset.py` | 从大语料中抽取随机子集并输出 parquet |
| 主流水线 | `preprocessor.py` | 规范化、分段、采样、拼音标注，写出训练/验证集 |
| 分词器 | `tokenizer.py` | 三词表分词器；编码/解码；拼音->汉字可能性掩码 |
| 词表资产 | `vocabs/` | chinese_vocab.txt、context_vocab.txt、pinyin_vocab.txt、config.yaml |
| 训练数据层 | `dataset.py` | 在线变换（片段选择、拼音增强）、NJT collate、流式数据集 |

`dataset.py` 在此处描述，因为它是数据表示层；训练部门消费它（见 `train.md`）。

## 2. `subset.py` — 语料子集抽取器

**功能：** 独立的辅助工具：从已有数据集（HF `save_to_disk` 目录或 parquet 文件/目录）中抽取随机子集并保存为 parquet 分片。用于构建较小的可用语料， 例如 `preprocessor.py` 消费的 fineweb 子集。

**用法：** 直接运行：`python subset.py`。配置通过模块常量： `INPUT_PATH`、`OUTPUT_PATH`、`INPUT_FORMAT`（`"hf"` 或 `"parquet"`）、 `SUBSET_SIZE`（比例或绝对行数）、`SEED`（默认 114514）、`NUM_SHARDS` （默认 20）。

### 函数

#### `_find_parquet_files(path)`
- 功能：查找路径下的 parquet 文件。
- 用法：由 `load_source` 调用；`path` 可以是单个 `.parquet` 文件或目录。
- 行为：若为以 `.parquet` 结尾的文件则返回 `[path]`；若为目录则返回递归 glob 到的 `*.parquet` 排序列表；路径不存在时返回 `[]`。

#### `load_source(path, fmt)`
- 功能：加载输入语料。
- 用法：`fmt="hf"` 用 `load_from_disk`；`fmt="parquet"` 用 `Dataset.from_parquet` 加载所有找到的 parquet 文件。
- 行为：找不到 parquet 时抛出 `FileNotFoundError`；未知格式抛出 `ValueError`。

#### `extract_subset(ds, subset_size, seed)`
- 功能：随机选择部分行。
- 用法：`subset_size` 为 (0, 1] 的比例或整数行数；`seed` 驱动随机数。
- 行为：计算 `n_keep`（限制在 `[0, n]`），用 numpy `default_rng` 置换行 索引，保留前 `n_keep` 个并排序（保证顺序回读高效），返回 `ds.select(...)`。

#### `save_parquet(ds, out_dir, num_shards)`
- 功能：将数据集写为 parquet，可分片。
- 用法：`num_shards <= 1` 时写单个 `data.parquet`。
- 行为：多分片时按连续分片写出 `data-<分片>-of-<总分片数>.parquet`； 必要时创建 `out_dir`。

#### `main()`
- 功能：编排 加载 -> 抽子集 -> 保存，并打印源/子集行数。

## 3. `preprocessor.py` — 主预处理流水线

**功能：** 核心预处理流水线：规范化原始文本，切成中文 / 停顿标点 / 非中文 三类连续片段，把片段切成限定长度的样本，为中文片段计算嵌套的逐字拼音， 切分训练/验证集，训练集以 MDS 并行写出，验证集以 HF Arrow 写出。

**用法：** 命令行两个开关：

- `python preprocessor.py --preprocess` — 运行完整流水线（所有数据源 -> 拼接 -> 字符统计 -> 训练/验证切分 -> MDS 训练集 -> Arrow 验证集原始版）。
- `python preprocessor.py --generate_val` — 读取 `val_original`，物化出 准备好的验证集（`prefix`/`suffix`/`pinyin` 三列）。

分词器由 `vocabs/config.yaml` 构建（`P2CTokenizer.from_config`）。

### 常量

- 数据源路径：`LCCC_PATH`、`MMC_PATH`、`CLUE_PATH`、`WIKIPEDIA_PATH`、 `ZHIHUKOL_PATH`、`FINEWEB_EDU_CHS_PATH`。生效的数据源由 `PARQUET_SOURCES`（zhihu-kol 的 INSTRUCTION/RESPONSE 列；fineweb 的 `text` 列被注释）、`JSONL_SOURCES`（LCCC、MMC）、`TEXTONLY_SOURCES` （CLUE）定义。
- 采样边界：`TRUNCATION_LEN_MIN=16`、`TRUNCATION_LEN_MAX=64`、 `MAX_TEXT_LEN=72`、`BAN_TEXT_LEN=96`、`MAX_NON_CHINESE_SEG_LEN=16`、 `MIN_CHINESE_RATIO=0.5`。
- 切分与输出：`SPLIT_RATIO=0.995`、训练/验证/原始输出目录、 `NUM_VAL_SHARDS=1`、`NUM_ORIGIN_SHARDS=15`、`SHUFFLE_SEED=114514`、 `NUM_PROC=15`、`BATCH_SIZE=8`。
- MDS：`MDS_COMPRESSION="zstd"`、`MDS_SIZE_LIMIT=1<<28`（约 256 MB/分片）、 列定义为 `text_list: json`、`pinyin_list: str`、`labels: json`。
- 验证集增强参数：`DROP_VOWELS=0.2`、`DROP_LAST_VOWEL=0.3`、 `VOWELS_DROPRATE_MIN=0.5`、`VOWELS_DROPRATE_MAX=1.0`、 `HETERONYM_CONFUSION=0.1`、`NO_CONTEXT_PROB=0.2`、`FORWARD_RATIO=0.0`。
- 片段标签：`PAUSE_LABEL=0`、`CHINESE_LABEL=1`、`NON_CHINESE_LABEL=2` （必须与 `dataset.py` 一致）。

### 辅助函数

#### `_get_initial_and_final(syllable)`
- 功能：把拼音音节拆成（声母，韵母）。
- 用法：内部使用；用于验证集拼音增强。
- 行为：先检查复合声母 `zh/ch/sh`，再检查单字符声母；找不到声母时返回 `("", syllable)`。

#### `_chinese_char_ratio(text)`
- 功能：字符串中落在 `[\u4e00-\u9fff]` 的字符占比。
- 用法：片段级与样本级过滤检查。
- 行为：空文本返回 `0.0`。

#### `_clean_emoji(text)`
- 功能：去除 `[...]` 括号内容（emoji 占位符）。
- 用法：`preprocess_text` 的第一步规范化。

#### `_remove_segmentation_spaces(text)`
- 功能：删除两个汉字之间的空格（修复已分词的语料）。
- 用法：第二步规范化。

#### `word_to_pinyin_nested(word)`
- 功能：把中文词转换为 `(clean_word, nested_pinyin)`，其中 `nested_pinyin` 对 `clean_word` 的每个字符恰好一个条目，每个条目是该 字符的可能读音列表，最可能读音排在首位。
- 用法：`_make_sample` 对每个中文片段调用。
- 行为：先对整词取最优（非多音）读音序列，再逐字取多音读音；过滤特殊的 `ê` 读音；存在最优读音时把它提到最前；无拼音的字符被丢弃，以保证字符 序列与拼音序列对齐。

#### `resolve_pinyin_list(pinyin_raw, heteronym_confusion_p=0.0)`
- 功能：把逐字读音列表压平为每个字一个读音。
- 用法：验证集增强使用；`dataset.py` 中同名函数用于训练。
- 行为：默认取每字第一个（最可能）读音；当 `heteronym_confusion_p > 0` 且某字有多个读音时，以该概率随机选一个非首位读音（仅训练期噪声）。

#### `build_val_pinyin(pinyin_flat_readings)`
- 功能：对验证集逐字读音列表做静态增强。
- 用法：由 `_prepare_val_batch` 调用。
- 行为：先做读音消歧（含多音混淆），再以概率 `DROP_VOWELS` 对整句增强： 每个既有声母又有韵母的音节以 `[VOWELS_DROPRATE_MIN, VOWELS_DROPRATE_MAX]` 区间内的均匀概率被替换成声母首字符；随后以概率 `DROP_LAST_VOWEL` 对最后 一个音节做同样替换。返回增强后的音节列表。

### 分段

#### `preprocess_text(raw_text, tokenizer)`
- 功能：规范化一条原始文本。
- 用法：`process_raw_text` 的第一阶段。
- 行为：去 emoji 占位符、删除汉字间空格、简繁转换 （`zhconv_rs.zhconv`，zh-hans）、NFKC 规范化，然后只保留 context 词表 内的字符外加停顿标点（`! ? 。 : ; ,` 与换行）；返回清洗后的字符串 （空输入返回空串）。

#### `_char_category(ch, tokenizer)`
- 功能：对单个字符分类。
- 行为：停顿标点返回 `PAUSE_LABEL`；在 chinese_vocab 中返回 `CHINESE_LABEL`；否则返回 `NON_CHINESE_LABEL`。

#### `segment_text(text, tokenizer, max_non_chinese_len=16)`
- 功能：把规范化文本切成 `(segments, labels)`。
- 用法：由 `process_raw_text` 调用。
- 行为：整条文本的中文占比低于 `MIN_CHINESE_RATIO` 时整条丢弃；把文本按 类别聚成最长连续段；停顿段折叠为第一个字符（一个 PAUSE 段）；中文段用 jieba 切成词段；非中文段仅在长度不超过 `max_non_chinese_len` 时保留。

### 采样

#### `build_samples(segs, labels)`
- 功能：把完整片段列表切成样本。
- 用法：由 `process_raw_text` 调用。
- 行为：随机取 `[16, 64]` 之间的 `trunc_len`，累计片段直到达到该长度；继续 延伸直到遇到"中文段之后的停顿段"为止（一旦超过 `MAX_TEXT_LEN=72`，则改为 在"中文段之后的非中文段"处停止）；若总长超过 `BAN_TEXT_LEN=96` 则丢弃该 窗口并从其后继续扫描；每个产出的窗口经 `_make_sample` 处理；产出后扫描 从停止处的停顿段之后继续。

#### `_make_sample(sample_segs, sample_labels)`
- 功能：把片段窗口打包成一个数据集样本。
- 行为：为每个中文片段计算嵌套拼音（整词无拼音的片段被丢弃）；至少要求有 一个中文片段；拼接后文本的中文占比低于 `MIN_CHINESE_RATIO` 时丢弃样本； 返回 `{"text_list", "pinyin_list"（JSON 字符串）, "labels"}`。

#### `process_raw_text(raw_text, tokenizer)`
- 功能：整条原始文本 -> 样本字典列表。
- 行为：先跑 `preprocess_text`；长度 ≤ 1 时返回 `[]`；然后分段并采样。

### 片段选择（离线验证集版本）

#### `select_span(labels, k, direction="backward")`
- 功能：把 1 起始的中文片段序号 `k` 映射为 `(sel, end, chinese_offset)`：`prefix = segments[0:sel]`、 `suffix = segments[sel:end]`（连续中文段），拼音切片为 `[chinese_offset : chinese_offset + (end - sel)]`。
- 用法：由 `_prepare_val_batch` 使用；`dataset.py` 中的训练期版本 `_select_span` 逻辑相同。
- 行为：`backward` 以第 k 个中文段为锚点向右延伸连续中文段；`forward` 以 锚点为结尾向左延伸；`bidirectional` 向两侧延伸；未知方向抛出 `ValueError`。

### 批处理与加载器

#### `process_pure_batch(batch, tokenizer)`
- 功能：把一批 `sentences`（原始文本列表）映射为样本列表。
- 行为：对每条句子、每个原始文本运行 `process_raw_text`，累加 `text_list`/`pinyin_list`/`labels`。

#### `process_parquet_batch(batch, tokenizer, text_columns)`
- 功能：与 `process_pure_batch` 类似，但读取 parquet 批的指定文本列。
- 行为：跳过缺失列与空字符串；累加所有产出样本。

#### `_find_files(path, suffix)`
- 功能：在目录下递归查找指定后缀的文件。
- 行为：返回排序列表；路径缺失或非目录时返回 `[]`。

#### `load_and_process_jsonl_source(tokenizer, jsonl_path)`
- 功能：加载并处理 JSONL 语料目录。
- 行为：查找 `*.jsonl` 文件；逐行解析 JSON——能解析为列表的行成为 `{"sentences": items}`（单对象行被丢弃，JSON 解析失败的行跳过）；构建 HF 数据集后用 `process_pure_batch` 映射（`batched=True`、`BATCH_SIZE=8`、 `NUM_PROC=15`）；返回处理后的数据集，无文件时返回 `None`。

#### `load_and_process_textonly_source(tokenizer, textonly_path)`
- 功能：加载并处理纯文本语料目录。
- 行为：查找 `*.txt` 文件；每个非空行成为一条句子；映射 `process_pure_batch`；返回处理后的数据集或 `None`。

#### `load_and_process_parquet_source(path, text_columns, tokenizer)`
- 功能：逐分片处理 parquet 语料。
- 行为：查找 `*.parquet` 文件并按随机顺序处理；每个分片只读取存在的文本 列，映射 `process_parquet_batch`（每个分片处理完即释放以控制内存），最后 拼接所有已处理分片；没有可用分片时返回 `None`。

### 切分与统计

#### `shuffled_train_test_split(ds, test_size, seed)`
- 功能：基于全量置换的训练/验证切分。
- 行为：用带种子的 numpy RNG 置换索引；置换结果的前 `n_test` 个索引归验证 集；两组索引都排序，保证顺序读回；经 `ds.select` 返回 `(train_ds, test_ds)`。

#### `count_characters(ds)`
- 功能：统计所有样本的总字符数。
- 行为：并行映射计数批处理，对每条 `text_list` 求各段长度之和再汇总。

### MDS 写出

#### `_write_mds_partition(args)`
- 功能：把数据集的一个连续分区写入 MDS 的 worker 入口。
- 用法：由 `save_train_as_mds` 在 `spawn` 进程池中执行；`args` 为 `(num_proc, shard_id, dataset_disk_path, out_dir)`。
- 行为：从磁盘 mmap 原始数据集，取第 `shard_id` 个连续分片，经 `MDSWriter` 以 zstd 压缩、256 MB 大小上限流式写出每个样本。

#### `save_train_as_mds(dataset, out_dir, num_proc=15)`
- 功能：以 MDS 并行保存训练集。
- 行为：先把完整数据集存到 `train_original`（15 分片），在 `num_proc` 个 worker 之间计算连续边界，spawn 进程池写出每个分区，再用 `streaming.base.util.merge_index`（`keep_local=True`）合并分区索引，使 结果成为单个有效的 MDS 目录。

### 验证集准备

#### `_prepare_val_batch(batch, tokenizer)`
- 功能：把一个验证样本物化为 `prefix` / `suffix` / `pinyin`。
- 行为：对每个样本随机选择中文片段序号 `k`；以概率 `NO_CONTEXT_PROB` 不使 用前缀（双向片段），否则以概率 `FORWARD_RATIO`（默认 0.0，即几乎总是 backward）选择 forward/backward 方向的片段；suffix 为连续中文片段； pinyin 列是 suffix 的增强后扁平音节列表的 JSON 字符串。

### `main()`
- 功能：命令行入口。
- 行为：`--preprocess` 时构建分词器，按序加载并处理所有数据源（JSONL、 纯文本、parquet），拼接，统计字符数，切分训练/验证，写出训练 MDS 与 `val_original` Arrow 数据集；`--generate_val` 时加载 `val_original`， 应用 `_prepare_val_batch`（批处理、单进程），保存 `val`（1 分片）。无任何 数据源时抛出 `RuntimeError`；结束时打印行数/字符数统计。

## 4. `tokenizer.py` — P2CTokenizer

**功能：** 跨部门使用的三词表分词器：

- **chinese_vocab** — post 模型的输出/目标空间（纯汉字）。
- **context_vocab** — pre 模型的输入空间（汉字 + 标点 + 其他符号）； special token 追加在末尾。
- **pinyin_vocab** — post 模型的输入空间（拼音音节）。

它还构建用于过滤 post 模型非法 logits 的拼音->汉字**可能性掩码**。

**用法：** `P2CTokenizer.from_config("vocabs/config.yaml")`；被 `preprocessor.py`（过滤）、`dataset.py`（编码）、`tasks/train.py`（词表大小 + 掩码）、`demo.py`、`export.py` 使用。

### `_read_vocab_tokens(path)`
- 功能：读取 unigram 词表文件（每行一个 token）。
- 行为：内容恰为 `\n` 两个字符的行会被转换为真正的换行 token；空行跳过； 按文件顺序返回 token 列表。

### `P2CTokenizer`

#### `__init__(chinese_vocab_path, context_vocab_path, pinyin_vocab_path, special_tokens_def=None)`
- 功能：构建三个词表。
- 行为：每个词表按文件顺序去重（先出现者获得最小 id）；special token 追加 在 context 词表末尾，id 从 `context_base_size` 起；其名字挂到 `self.spec_tokens` 上（例如 `tokenizer.spec_tokens.bos_token`）。

#### `from_config(config_path)` / `from_config_dict(config_dir, config_dict)`
- 功能：按 `vocabs/config.yaml` 结构（`{"vocabs": {chinese_vocab, context_vocab, pinyin_vocab, context_special_tokens}}`）构建分词器的 classmethod。
- 行为：词表路径相对配置文件所在目录解析；`context_special_tokens` 可选 （缺省为 None）。

#### 属性
- `chinese_vocab_size`、`pinyin_vocab_size`、`context_vocab_size`、 `vocab_size`（context 大小的别名，pre 模型使用）、`proj_size` （chinese 大小的别名，post 模型输出）、`num_special_tokens`。

#### 编码
- `encode_context(text)` — 按 context_vocab 映射字符；词表外的字符静默丢 弃；返回 id 列表（用于 pre 模型输入与前缀）。
- `encode_chinese(text)` — 按 chinese_vocab 映射字符，丢弃未知字符；用于 post 模型目标。
- `encode_pinyin(pinyin_list)` — 按 pinyin_vocab 映射音节；词表外的音节 映射到编辑距离（`_edit_distance`）最小的词表内音节。

#### 检查与过滤
- `check_pinyin(pinyin_list)` — 当且仅当所有音节都在 pinyin 词表中时返回 True。
- `filter_text(text, allow_characters=None, vocab="chinese")` — 只保留所选 词表内的字符，外加 `allow_characters`；`vocab="context"` 使用更宽的 context 词表。
- `is_chinese(ch)` — 当且仅当 `ch` 是 chinese_vocab 中的单个字符。
- `check(text, vocab="chinese")` — 当且仅当 `text` 每个字符都在所选词表中。

#### 解码
- `ids_to_text(ids)` — chinese_vocab id -> 字符。
- `ids_to_context_text(ids)` — context_vocab id -> 字符，跳过 special token （id ≥ context_base_size）。
- `decode_greedy(ids)` — `ids_to_text` 的别名。

#### `_edit_distance(a, b)`（静态）
- 功能：Levenshtein 编辑距离。
- 用法：未知拼音音节的回退映射。
- 行为：完整 DP 表；O(m·n)。

#### `create_possibility_map()`
- 功能：构建 `(pinyin_vocab_size, chinese_vocab_size)` 布尔掩码；条目 `(p, c)` 为 True 当且仅当拼音 `p` 可能对应汉字 `c`。
- 用法：训练器把该掩码赋给 `post_model.logits_mask`；post 模型把掩码为 False 的所有 logits 置为 `-inf`。
- 行为：对每个汉字用 pypinyin 取全部多音读音（`heteronym=True`、 `strict=False`、`errors="ignore"`、`Style.NORMAL`、`v_to_u=False`）；跳过 罕见的 `ê` 读音；标记完整音节精确匹配，也标记长度为 1–3、且为合法简拼 声母并在 pinyin 词表中的前缀（因此输入 `zh` 也能到达完整拼音以 `zh` 开头的汉字）。完全无拼音的字符打印警告。

## 5. `vocabs/` — 词表资产

- `config.yaml` — 映射 `vocabs.chinese_vocab`、`vocabs.context_vocab`、 `vocabs.pinyin_vocab` 和 `vocabs.context_special_tokens`（此处为 `bos_token`）；由 `P2CTokenizer.from_config` 消费。
- `chinese_vocab.txt` — 每行一个汉字（8104 行）；定义 post 模型输出空间与 "是否为汉字"的判断。
- `context_vocab.txt` — 每行一个 token（8223 行）；pre 模型输入空间（汉字 + 标点 + 符号）。
- `pinyin_vocab.txt` — 每行一个拼音音节（443 行）；post 模型输入空间。

行内写 `\n` 两个字面字符表示真正的换行 token（见 `_read_vocab_tokens`）。

## 6. `dataset.py` — 训练数据层

**功能：** 训练期的数据表示：在线样本变换（片段选择 + 拼音增强）、NJT collate 函数、以及 MDS 训练集的流式数据集封装。它也复用了 `preprocessor.py` 的分段标签常量。

**用法：** 被 `tasks/train.py` 用于变换、collate、流式训练集和验证数据集加载。

### 常量
`PAUSE_LABEL=0`、`CHINESE_LABEL=1`、`NON_CHINESE_LABEL=2` — 必须与 `preprocessor.py` 一致。拼音增强集合：`_COMP_CONSONANTS`（`zh/ch/sh`）、 `_ALL_INITIALS`（全部简拼声母）。

### 拼音增强辅助函数

#### `_get_initial_and_final(syllable)`
- 功能：把音节拆成（声母，韵母）。
- 行为：先复合声母后单字符声母；无声母时返回 `("", syllable)`。

#### `augment_pinyin_sequence(pinyin_list, aug_cfg)`
- 功能：对音节列表做丢韵母增强。
- 用法：由 `transform_pinyin_predict_train` 调用；由 `dataset.augmentation` 配置块配置。
- 行为：以概率 `drop_vowels` 对整句增强：每个既有声母又有韵母的音节，以 `vowels_droprate` 区间内均匀概率被替换为声母首字符（例如 `zhu -> z`）； 独立地以概率 `drop_last_vowel` 对最后一个音节做同样替换。

#### `resolve_pinyin_list(pinyin_raw, heteronym_confusion_p=0.0)`
- 功能：压平逐字读音列表（与 `preprocessor.py` 语义相同）。
- 行为：默认取首读音；`heteronym_confusion_p > 0` 时可能随机取非首读音。

### 黄金序列片段选择

#### `PHI`、`_compute_base_seed_norm(text)`
- 功能：由文本 MD5 导出的确定性种子，落在 [0, 1)。
- 行为：MD5 前 8 字节按大端整数除以 2^64。

#### `_pick_golden_choice(count_chinese, text, epoch)`
- 功能：`golden-sequence` 在线策略的确定性黄金比例片段选择。
- 用法：由 `transform_pinyin_predict_train` 调用；返回值是 `[1, count_chinese]` 内 1 起始的序号。
- 行为：计算 `relative = (base + epoch·φ) mod 1`，其中 φ 为黄金比例；映射 到中文片段序号。相同的（text, epoch）总是得到相同的选择，跨轮次均匀铺开 覆盖。

#### `_select_span(labels, k, direction="backward")`
- 功能：1 起始中文片段序号 -> `(sel, end, chinese_offset)`（与 `preprocessor.select_span` 逻辑一致）。
- 行为：`backward`（向右延伸）、`forward`（向左延伸）、`bidirectional` （两侧）；未知方向抛出 `ValueError`。

### 变换

#### `transform_pinyin_predict_train(batch, tokenizer, aug_cfg=None, online_policy=None, epoch=None)`
- 功能：把一批预处理样本转换为 `prefix_ids` / `postfix_ids` / `target_ids` 列表（及长度）。
- 用法：训练变换；在 `P2CStreamingDataset` 内部与训练器每批调用。
- 行为：解析 JSON 拼音；跳过无中文样本；按 `golden-sequence`（依赖 epoch， 确定性）或 `random` 策略选择片段序号 `k`；以 `no_context_prob` 概率使用 空前缀（双向片段），否则以概率 `forward_prob`（未配置时默认 0.5）选择 forward/backward 片段；前缀为 `[bos_token] + encode_context(prefix_text)`； suffix 拼音为片段范围内逐字读音的扁平序列，先消歧（多音混淆）再增强 （丢韵母）；`postfix_ids` 用 `encode_pinyin`，`target_ids` 用 `encode_chinese`。无中文的样本整体跳过。

#### `transform_pinyin_predict_val(batch, tokenizer, aug_cfg=None)`
- 功能：把准备好的验证批（`prefix` / `suffix` / `pinyin` 列）转换为 id。
- 行为：前缀为 `[bos_token] + encode_context(prefix_text)`；postfix 用 `encode_pinyin`；target 用 `encode_chinese`；不做在线增强（已在离线阶段 物化）。

### Collate

#### `make_collate_fn()`
- 功能：构建 NJT collate 函数。
- 用法：传给 `StreamingDataLoader` 与验证 `DataLoader`。
- 行为：把 `prefix_ids` / `postfix_ids` / `target_ids` 各自转换为锯齿嵌套 张量（`torch.nested.nested_tensor(..., layout=jagged)`）；返回以 `prefix_ids_njt`、`postfix_ids_njt`、`target_ids_njt` 为键的批次字典。

### 加载

#### `create_dataset(data_path, keep_in_memory=False)`
- 功能：从磁盘加载 HF 数据集目录。
- 行为：`load_from_disk(data_path, keep_in_memory=keep_in_memory)`。

### `P2CStreamingDataset(StreamingDataset)`
- 功能：MDS 训练集的流式封装，逐条应用在线训练变换。
- 用法：训练器用 `local`、`tokenizer`、`aug_cfg`、`online_policy` 及流式 参数（shuffle、cache_limit 等）构造。
- 行为：`__getitem__` 把一条原始 MDS 样本包装成单元素批次，以 `epoch=self.next_epoch - 1` 调用 `transform_pinyin_predict_train` （即黄金序列种子从 epoch 0 开始），返回各结果列表的第一个元素。
