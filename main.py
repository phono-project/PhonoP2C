import os

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    os.environ["WANDB_CONSOLE"] = "off"
    if cfg.system.expandable_segments_enabled:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch
    import logging
    from utils.distributed import destroy_distributed, initialize_distributed

    task_name = cfg.task.get("task_type", "train")
    distributed = initialize_distributed(cfg.system) if task_name == "train" else None
    if distributed is not None:
        cfg.system.device = str(distributed.device)

    log_level = cfg.output.logging.log_level.upper()
    if log_level != "DEFAULT":
        numeric_level = getattr(logging, log_level, logging.WARNING)
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            force=True,
        )
        logging.getLogger("hydra").setLevel(numeric_level)
        logging.getLogger("torch").setLevel(numeric_level)

    if cfg.system.tf32_enabled:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if cfg.system.get("set_seed", False):
        import random
        import numpy as np

        seed = int(cfg.system.get("seed", 42))
        if distributed is not None:
            seed += distributed.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    torch.set_float32_matmul_precision('high')

    # Currently, FX graph cache is not supported when using Nested Tensor.
    torch._inductor.config.fx_graph_cache = False

    # Dispatch to the appropriate task handler.
    try:
        if task_name == "preprocess":
            from omegaconf import OmegaConf
            from datasets_pipeline.preprocessor import run_preprocess

            dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
            run_preprocess(dataset_cfg, generate_val=cfg.task.get("generate_val", True))
        elif task_name == "export":
            from export import run_export

            run_export(cfg.task)
        else:
            from tasks.train import Trainer
            trainer = Trainer(cfg, distributed)
            trainer.train()
    finally:
        if distributed is not None:
            destroy_distributed()


if __name__ == "__main__":
    main()
