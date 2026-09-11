from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from export.task import DTYPES, load_model_from_checkpoint, run_export


def _export_config():
    config_path = Path(__file__).parents[1] / "config" / "task" / "export.yaml"
    return OmegaConf.load(config_path)


def test_export_config_exposes_inputs_outputs_and_quantization():
    cfg = _export_config()

    assert cfg.task_type == "export"
    assert cfg.checkpoint_dir
    assert cfg.output_dir
    assert cfg.outputs.pre_model.endswith(".pte")
    assert cfg.outputs.post_model.endswith(".pte")
    assert cfg.quantization.mode in {"none", "w8a8", "w4a8"}
    assert cfg.dtype in DTYPES


def test_export_rejects_non_cpu_device_before_loading_checkpoint():
    cfg = _export_config()
    cfg.device = "cuda"

    with pytest.raises(ValueError, match="requires task.device=cpu"):
        run_export(cfg)


def test_checkpoint_loader_requires_both_models(tmp_path):
    (tmp_path / "pre_model").mkdir()

    with pytest.raises(FileNotFoundError, match="pre_model/ and post_model/"):
        load_model_from_checkpoint(tmp_path, torch.device("cpu"))
