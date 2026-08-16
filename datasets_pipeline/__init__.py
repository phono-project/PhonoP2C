"""PhonoP2C datasets & preprocessing pipeline.

Shared components:
    constants.py  — segment labels / punctuation sets
    pinyin.py     — pinyin splitting, augmentation, heteronym sampling
    segments.py   — span selection & Chinese-segment char offsets
    dataset.py    — training-time transforms, NJT collate, streaming dataset
    preprocessor.py — corpus -> MDS/Arrow pipeline (config-driven)
"""

from datasets_pipeline.constants import (
    CHINESE_LABEL,
    NON_CHINESE_LABEL,
    PAUSE_LABEL,
)
from datasets_pipeline.dataset import (
    P2CStreamingDataset,
    create_dataset,
    make_collate_fn,
    transform_pinyin_predict_train,
    transform_pinyin_predict_val,
)
from datasets_pipeline.preprocessor import run_preprocess
from datasets_pipeline.pinyin import (
    augment_pinyin_sequence,
    get_initial_and_final,
    sample_pinyin_list,
)
from datasets_pipeline.segments import chinese_segment_char_prefixes, select_span

__all__ = [
    "CHINESE_LABEL", "NON_CHINESE_LABEL", "PAUSE_LABEL",
    "P2CStreamingDataset", "create_dataset", "make_collate_fn",
    "transform_pinyin_predict_train", "transform_pinyin_predict_val",
    "run_preprocess",
    "augment_pinyin_sequence", "get_initial_and_final", "sample_pinyin_list",
    "chinese_segment_char_prefixes", "select_span",
]
