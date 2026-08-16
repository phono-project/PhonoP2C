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
        seed = cfg.system.get("seed", 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    torch.set_float32_matmul_precision('high')

    # Currently, FX graph cache is not supported when using Nested Tensor.
    torch._inductor.config.fx_graph_cache = False

    # Dispatch to the appropriate task handler
    task_name = cfg.task.get("task_type", "train")

    if task_name == "preprocess":
        from omegaconf import OmegaConf
        from datasets_pipeline.preprocessor import run_preprocess

        dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
        run_preprocess(dataset_cfg, generate_val=cfg.task.get("generate_val", True))
    elif task_name == "param_search":
        from tasks.param_search import ParamSearchRunner
        runner = ParamSearchRunner(cfg)
        runner.run()
    else:
        from tasks.train import Trainer
        trainer = Trainer(cfg)
        trainer.train()


if __name__ == "__main__":
    main()
