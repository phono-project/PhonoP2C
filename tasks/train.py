"""
PostfixLM training loop (new-standard encoder-decoder architecture).

Trains both PhonoP2CPreModel (causal decoder) and PhonoP2CPostModel (pinyin
encoder) jointly using NJT batches.  The whole two-pass forward (unconditional
pass + conditional cross-attended pass) is wrapped in PhonoP2CTrainWrapper and
torch.compile is applied to the wrapper.  The two losses (unconditional /
conditional) are logged separately; pre and post models are saved
individually.
"""

import os
import logging

import torch
import torch.distributed as dist
import bitsandbytes as bnb
from transformers import get_wsd_schedule
import wandb
from streaming import StreamingDataLoader
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
)

from datasets_pipeline import (
    make_collate_fn,
    create_dataset,
    transform_pinyin_predict_val,
    P2CStreamingDataset,
)
from model.config import build_configs_from_dict
from model.model import PhonoP2CPreModel, PhonoP2CPostModel
from model.wrapper import PhonoP2CTrainWrapper
from model.beam_search import beam_search_batch
from tokenizer import P2CTokenizer

from torchao.float8 import Float8LinearConfig, convert_to_float8_training
from utils.float8 import module_filter_fn
from loss import get_loss_fn
from metrics import MetricsAccumulator, TopKSentenceAccuracy
from utils.distributed import (
    DistributedContext,
    DistributedEvalSampler,
    ddp_local_mean_scale,
)

from rich.progress import ProgressColumn
from rich.text import Text


class IterationSpeedColumn(ProgressColumn):
    def render(self, task):
        speed = task.speed
        if speed is None:
            return Text("? it/s", style="bold blue")
        return Text(f"{speed:.2f} it/s", style="bold blue")


class MofNCompleteColumn(ProgressColumn):
    def render(self, task):
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else "?"
        return Text(f"{completed}/{total}", style="grey50")


def _build_progress(cfg: DictConfig, disable: bool = False) -> Progress:
    pcfg = cfg.output.progress_bar
    columns = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=pcfg.get("bar_width", None)),
        TaskProgressColumn(),
    ]
    if pcfg.get("show_time_elapsed", True):
        columns.append(TimeElapsedColumn())

    if pcfg.get("show_time_remaining", True):
        columns.append(TextColumn("•"))
        columns.append(TimeRemainingColumn())

    if pcfg.get("show_ratio", True):
        columns.append(TextColumn("•"))
        columns.append(MofNCompleteColumn())

    if pcfg.get("show_speed", True):
        columns.append(TextColumn("•"))
        columns.append(IterationSpeedColumn())

    columns.append(TextColumn("{task.fields[postfix]}"))
    return Progress(
        *columns,
        refresh_per_second=pcfg.get("refresh_per_second", 10),
        transient=pcfg.get("transient", True),
        disable=disable,
    )


def build_model_configs(cfg: DictConfig, tokenizer: P2CTokenizer) -> tuple:
    """Build PreModelConfig and PostModelConfig from YAML, filling vocab sizes."""
    d = OmegaConf.to_container(cfg.model, resolve=True)
    d = dict(d) if d else {}

    vocab_sizes = {
        "context": tokenizer.context_vocab_size,
        "pinyin": tokenizer.pinyin_vocab_size,
        "chinese": tokenizer.chinese_vocab_size,
    }
    return build_configs_from_dict(d, vocab_sizes)


class Trainer:
    def __init__(self, cfg: DictConfig, distributed: DistributedContext | None = None):
        self.cfg = cfg
        self.distributed = distributed or DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device(cfg.system.device),
        )
        self.device = self.distributed.device
        self.is_main = self.distributed.is_main
        if self.is_main and self.distributed.is_distributed:
            print(
                f"DDP enabled: world_size={self.distributed.world_size}, "
                f"device={self.device}, per-rank batch_size={cfg.task.batchsize}, "
                f"global batch_size={cfg.task.batchsize * self.distributed.world_size}"
            )

        # Logging
        log_level = cfg.output.logging.get("log_level", "WARNING").upper()
        logging.basicConfig(
            level=getattr(logging, log_level, logging.WARNING),
            format="%(asctime)s [%(levelname)s] %(message)s",
        )

        # Tokenizer
        self.tokenizer = P2CTokenizer.from_config(cfg.dataset.vocabs_config)

        # Model config
        self.pre_cfg, self.post_cfg = build_model_configs(cfg, self.tokenizer)

        # Build logits mask
        if self.is_main:
            print("Building pinyin->Chinese possibility map...")
        logits_mask = self.tokenizer.create_possibility_map().to(self.device)
        if self.is_main:
            print(f"  Mask: {logits_mask.shape}, nonzero={logits_mask.sum().item()} "
                  f"({100 * logits_mask.sum().item() / logits_mask.numel():.1f}%)")

        # Build models
        self.pre_model = PhonoP2CPreModel(self.pre_cfg).to(self.device)

        self.post_model = PhonoP2CPostModel(self.post_cfg).to(self.device)
        self.post_model.logits_mask = logits_mask

        load_from = cfg.task.get("load_from")
        if load_from:
            pre_path = os.path.join(load_from, "pre_model")
            post_path = os.path.join(load_from, "post_model")
            if not os.path.isdir(pre_path) or not os.path.isdir(post_path):
                raise FileNotFoundError(
                    f"Checkpoint {load_from!r} must contain pre_model/ and post_model/"
                )
            self.pre_model = PhonoP2CPreModel.from_pretrained(pre_path).to(self.device)
            self.post_model = PhonoP2CPostModel.from_pretrained(post_path).to(self.device)
            self.post_model.logits_mask = logits_mask
            if self.is_main:
                print(f"Loaded model parameters from {load_from}")

        total_params = (sum(p.numel() for p in self.pre_model.parameters()) +
                        sum(p.numel() for p in self.post_model.parameters()))
        if self.is_main:
            print(f"Pre model:  {sum(p.numel() for p in self.pre_model.parameters()) / 1e6:.2f}M params")
            print(f"Post model: {sum(p.numel() for p in self.post_model.parameters()) / 1e6:.2f}M params")
            print(f"Total:      {total_params / 1e6:.2f}M params")

        # Float8 acceleration
        if cfg.system.ao_acceleration == 'float8':
            fp8_config = Float8LinearConfig(pad_inner_dim=True)
            self.pre_model = convert_to_float8_training(self.pre_model, module_filter_fn=module_filter_fn, config=fp8_config)
            self.post_model = convert_to_float8_training(self.post_model, module_filter_fn=module_filter_fn, config=fp8_config)

        # Gradient checkpointing: recompute layer activations in backward to
        # cut peak activation memory, letting a larger batchsize fill the ALUs.
        if cfg.system.get("gradient_checkpointing", False) and not self.distributed.is_distributed:
            self.pre_model.gradient_checkpointing = True
            self.post_model.gradient_checkpointing = True
        elif cfg.system.get("gradient_checkpointing", False) and self.is_main:
            print(
                "WARNING: gradient checkpointing is disabled under DDP: PyTorch 2.13 "
                "currently gives NJT ragged dimensions different symbolic IDs "
                "during checkpoint recomputation. This changes memory/compute "
                "usage only; the model and loss are unchanged."
            )

        # Loss function (delivered into the wrapper so the loss is computed
        # inside the compiled region for better optimization).
        self.loss_type = cfg.task.get("loss_type", "ce")
        self.loss_fn = get_loss_fn(
            loss_type=self.loss_type,
            focal_loss_alpha=cfg.task.get("focal_loss_alpha", 0.25),
            focal_loss_gamma=cfg.task.get("focal_loss_gamma", 1.0),
            label_smoothing_epsilon=cfg.task.get("label_smoothing_epsilon", None),
        )

        # Weight of the unconditional loss in the total loss.  A value of 0.0
        # disables it entirely (skipping its computation and wandb logging).
        self.unconditional_loss_lambda = cfg.task.get("unconditional_loss_lambda", 1.0)

        # Two-pass training wrapper
        self.model = PhonoP2CTrainWrapper(
            self.pre_model, self.post_model, self.loss_fn, self.unconditional_loss_lambda
        )

        # Compile the wrapped forward pass
        if cfg.task.compile_model:
            mode = cfg.task.get("compile_mode", "default")
            import torch._dynamo.config as dynamo_config
            dynamo_config.capture_scalar_outputs = True
            dynamo_config.capture_dynamic_output_shape_ops = True
            self.model = torch.compile(self.model, mode=mode, dynamic=True)

        if self.distributed.is_distributed:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.device.index],
                output_device=self.device.index,
                broadcast_buffers=False,
            )

        # Logging
        self.log_cfg = cfg.logging.train
        self.checkpoint_dir = self.log_cfg.checkpoint_dir
        if self.is_main:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.distributed.barrier()
        if self.log_cfg.log_with_wandb and self.is_main:
            wandb.init(
                project=self.log_cfg.project_name,
                name=self.log_cfg.run_name,
                notes=self.log_cfg.get("notes", ""),
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            wandb.watch(self.model, log="all", log_freq=cfg.logging.train.histogram_interval)

        # Data pipeline
        self.online_policy = OmegaConf.to_container(cfg.dataset.online_policy, resolve=True) if cfg.dataset.get("online_policy") else {}
        self.aug_cfg = OmegaConf.to_container(cfg.dataset.augmentation, resolve=True) if cfg.dataset.get("augmentation") else {}

        self.train_ds = P2CStreamingDataset(
            local=cfg.dataset.train_dir_mds,
            tokenizer=self.tokenizer,
            aug_cfg=self.aug_cfg,
            online_policy=self.online_policy,
            shuffle=True,
            shuffle_algo=cfg.dataset.shuffle.algo,
            shuffle_block_size=cfg.dataset.shuffle.blocksize,
            cache_limit=cfg.dataset.shuffle.cache_limit,
            batch_size=cfg.task.batchsize,
        )
        self.train_loader = StreamingDataLoader(
            self.train_ds,
            batch_size=cfg.task.batchsize,
            num_workers=cfg.system.num_workers,
            collate_fn=make_collate_fn(),
        )

        self.epoch_steps = len(self.train_loader)
        self.total_steps = self.epoch_steps * cfg.task.epochs
        if cfg.task.get("warmup_steps") is not None:
            warmup_steps = cfg.task.warmup_steps
        elif cfg.task.get("warmup_ratio") is not None:
            warmup_steps = int(self.total_steps * cfg.task.warmup_ratio)
        else:
            warmup_steps = 0

        # Validation dataset
        self.val_ds = create_dataset(cfg.dataset.val_dir, keep_in_memory=cfg.system.keep_in_memory)
        self.val_ds.set_transform(
            lambda batch: transform_pinyin_predict_val(batch, self.tokenizer, None)
        )
        self.val_sampler = DistributedEvalSampler(
            self.val_ds,
            rank=self.distributed.rank,
            world_size=self.distributed.world_size,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=cfg.task.batchsize,
            sampler=self.val_sampler,
            pin_memory=True,
            collate_fn=make_collate_fn(),
            num_workers=cfg.system.num_workers,
        )

        # Collect all parameters from both models
        def get_optimizer_params(model, weight_decay):
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear, )
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.RMSNorm, torch.nn.Embedding)

            for mn, m in model.named_modules():
                for pn, p in m.named_parameters():
                    fpn = f"{mn}.{pn}" if mn else pn

                    if pn.endswith('bias'):
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        no_decay.add(fpn)

            param_dict = {pn: p for pn, p in model.named_parameters()}
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, f"Parameters {str(inter_params)} made it into both decay/no_decay sets!"
            assert len(param_dict.keys() - union_params) == 0, f"Parameters {str(param_dict.keys() - union_params)} were not separated!"

            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            return optim_groups

        param_groups = get_optimizer_params(self.pre_model, cfg.task.weight_decay) + \
                       get_optimizer_params(self.post_model, cfg.task.weight_decay)

        if cfg.task.optimizer == "adamw":
            if cfg.system.optim_8bit:
                if cfg.system.optim_paged:
                    self.optim = bnb.optim.PagedAdamW8bit(
                        param_groups,
                        lr=cfg.task.max_learning_rate,
                        weight_decay=cfg.task.weight_decay,
                        betas=tuple(cfg.task.betas),
                    )
                else:
                    self.optim = bnb.optim.AdamW8bit(
                        param_groups,
                        lr=cfg.task.max_learning_rate,
                        weight_decay=cfg.task.weight_decay,
                        betas=tuple(cfg.task.betas),
                    )
            else:
                if cfg.system.optim_paged:
                    self.optim = bnb.optim.PagedAdamW32bit(
                        param_groups,
                        lr=cfg.task.max_learning_rate,
                        weight_decay=cfg.task.weight_decay,
                        betas=tuple(cfg.task.betas),
                    )
                else:
                    self.optim = bnb.optim.AdamW32bit(
                        param_groups,
                        lr=cfg.task.max_learning_rate,
                        weight_decay=cfg.task.weight_decay,
                        betas=tuple(cfg.task.betas),
                    )
        elif cfg.task.optimizer == "ademamix":
            if cfg.system.optim_8bit:
                if cfg.system.optim_paged:
                    self.optim = bnb.optim.PagedAdEMAMix8bit(param_groups, lr=cfg.task.max_learning_rate, weight_decay=cfg.task.weight_decay)
                else:
                    self.optim = bnb.optim.PagedAdEMAMix8bit(param_groups, lr=cfg.task.max_learning_rate, weight_decay=cfg.task.weight_decay)
            else:
                if cfg.system.optim_paged:
                    self.optim = bnb.optim.PagedAdEMAMix32bit(param_groups, lr=cfg.task.max_learning_rate, weight_decay=cfg.task.weight_decay)
                else:
                    self.optim = bnb.optim.PagedAdEMAMix32bit(param_groups, lr=cfg.task.max_learning_rate, weight_decay=cfg.task.weight_decay)

        self.schd = get_wsd_schedule(
            self.optim,
            num_warmup_steps=warmup_steps,
            num_decay_steps=cfg.task.decay_steps,
            num_training_steps=self.total_steps,
            min_lr_ratio=cfg.task.min_learning_rate / cfg.task.max_learning_rate,
        )

        if load_from:
            optimizer_path = os.path.join(load_from, "optim_state", "optimizer.pt")
            if not os.path.isfile(optimizer_path):
                raise FileNotFoundError(f"Checkpoint {load_from!r} is missing optim_state/optimizer.pt")
            self.optim.load_state_dict(torch.load(optimizer_path, map_location=self.device))

            scheduler_path = os.path.join(load_from, "optim_state", "scheduler.pt")
            if os.path.isfile(scheduler_path):
                self.schd.load_state_dict(torch.load(scheduler_path, map_location=self.device))
            if self.is_main:
                print(f"Loaded optimizer state from {load_from}")
        self.use_amp = cfg.system.mixed_precision == "bf16"

        # Beam search metric settings
        self.beam_width = cfg.task.get("beam_width", 3)
        # Number of samples decoded per batched beam-search call (controls the
        # KV-cache memory of the decode batch, B * beam_width).
        self.beam_chunk_size = cfg.task.get("beam_chunk_size", 64)
        # Strided sampling rate for the beam-search metric: decode every
        # stride-th sample (in dataset order) to cut computation while keeping
        # an even, unbiased spread across sources/classes.
        self.beam_stride = max(1, int(cfg.task.get("beam_search_stride", 4)))

    def _save_checkpoint(self, save_path: str) -> None:
        if self.is_main:
            pre_to_save = self.pre_model._orig_mod if hasattr(self.pre_model, "_orig_mod") else self.pre_model
            post_to_save = self.post_model._orig_mod if hasattr(self.post_model, "_orig_mod") else self.post_model

            pre_to_save.save_pretrained(os.path.join(save_path, "pre_model"), safe_serialization=True)
            post_to_save.save_pretrained(os.path.join(save_path, "post_model"), safe_serialization=True)

            optim_path = os.path.join(save_path, "optim_state")
            os.makedirs(optim_path, exist_ok=True)
            torch.save(self.optim.state_dict(), os.path.join(optim_path, "optimizer.pt"))
            torch.save(self.schd.state_dict(), os.path.join(optim_path, "scheduler.pt"))

            config_path = os.path.join(save_path, "configs")
            os.makedirs(config_path, exist_ok=True)
            OmegaConf.save(self.cfg, os.path.join(config_path, "config.yaml"), resolve=True)
        self.distributed.barrier()

    def _forward_batch(self, batch, model=None):
        prefix_njt = batch["prefix_ids_njt"].to(self.device)
        suffix_njt = batch["suffix_ids_njt"].to(self.device)
        uncond_target_njt = batch["uncond_target_ids_njt"].to(self.device)
        postfix_njt = batch["postfix_ids_njt"].to(self.device)
        target_njt = batch["target_ids_njt"].to(self.device)
        prefix_lens = batch["prefix_lengths"].to(self.device)

        flat_prefix = prefix_njt.values()
        prefix_offsets = prefix_njt.offsets()
        flat_suffix = suffix_njt.values()
        suffix_offsets = suffix_njt.offsets()
        flat_uncond_target = uncond_target_njt.values()
        flat_postfix = postfix_njt.values()
        flat_target_ids = target_njt.values()

        prefix_seq_lens = prefix_offsets[1:] - prefix_offsets[:-1]
        min_sl_prefix = prefix_seq_lens.min().item()
        max_sl_prefix = prefix_seq_lens.max().item()

        suffix_seq_lens = suffix_offsets[1:] - suffix_offsets[:-1]
        min_sl_suffix = suffix_seq_lens.min().item()
        max_sl_suffix = suffix_seq_lens.max().item()

        full_lens = prefix_lens + suffix_seq_lens
        min_sl_full = full_lens.min().item()
        max_sl_full = full_lens.max().item()

        forward_model = self.model if model is None else model
        out = forward_model(
            flat_prefix, prefix_offsets, flat_suffix, suffix_offsets,
            flat_postfix, flat_uncond_target, flat_target_ids, prefix_lens,
            min_sl_prefix, max_sl_prefix, min_sl_suffix, max_sl_suffix,
            min_sl_full, max_sl_full,
        )

        return out, flat_target_ids, suffix_offsets, flat_uncond_target

    def validate(self, model, loader, epoch, global_step, progress):
        model.eval()
        metrics_acc = MetricsAccumulator(ece_bins=self.cfg.task.ece_bins, ece_top_k=self.cfg.task.ece_top_k)
        beam_acc = TopKSentenceAccuracy(k=self.beam_width)
        val_cond_loss_sum = 0.0
        val_cond_tokens = 0
        val_uncond_loss_sum = 0.0
        val_uncond_tokens = 0

        pre_model = self.pre_model
        post_model = self.post_model

        # Collect only this rank's share of the global strided beam subset.
        beam_samples: list[tuple] = []
        local_sample_index = 0

        val_task = progress.add_task(
            f"[cyan]Validating Epoch {epoch + 1}/{self.cfg.task.epochs}",
            total=len(loader),
            postfix="",
        )

        with torch.no_grad():
            for batch in loader:
                with torch.amp.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                    out, flat_target_ids, target_offsets, flat_uncond_target = self._forward_batch(
                        batch, model=model
                    )

                cond_tokens = flat_target_ids.numel()
                val_cond_loss_sum += out.conditional_loss.item() * cond_tokens
                val_cond_tokens += cond_tokens
                if out.unconditional_loss is not None:
                    uncond_tokens = int((flat_uncond_target != -100).sum().item())
                    val_uncond_loss_sum += out.unconditional_loss.item() * uncond_tokens
                    val_uncond_tokens += uncond_tokens

                # Update metrics (conditional logits are the predictions)
                metrics_acc.update(out.conditional_logits.detach(), flat_target_ids, target_offsets)

                # Use the CPU batch so deferred beam search does not retain
                # every validation target (and its backing storage) in VRAM.
                # Stride each rank's interleaved partition locally so beam
                # work stays balanced even when stride and world size overlap.
                for prefix_t, postfix_t, target_t in zip(
                    batch["full_prefix_ids_njt"].unbind(),
                    batch["postfix_ids_njt"].unbind(),
                    batch["target_ids_njt"].unbind(),
                ):
                    if local_sample_index % self.beam_stride == 0:
                        beam_samples.append(
                            (prefix_t.tolist(), postfix_t.tolist(), target_t.tolist())
                        )
                    local_sample_index += 1

                progress.update(val_task, advance=1, postfix=f"[red]loss: {out.loss.item():.4f}")

        # Group by pinyin length so each beam-search batch is rectangular.
        beam_groups: dict[int, list] = {}
        for prefix, pinyin, target in beam_samples:
            beam_groups.setdefault(len(pinyin), []).append((prefix, pinyin, target))

        # Standalone progress bar for the beam-search metric.
        beam_dtype = torch.bfloat16 if self.use_amp else None
        beam_task = progress.add_task(
            f"[cyan]S-ACC@{self.beam_width}-beam",
            total=len(beam_samples),
            postfix=f"[red]{len(beam_samples)} local samples (1/{self.beam_stride})",
        )
        for group in beam_groups.values():
            for i in range(0, len(group), self.beam_chunk_size):
                chunk = group[i:i + self.beam_chunk_size]
                prefixes = [c[0] for c in chunk]
                pinyins = [c[1] for c in chunk]
                targets = torch.tensor([c[2] for c in chunk], device=self.device)
                with torch.amp.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                    _, beam_ids = beam_search_batch(
                        pre_model, post_model, prefixes, pinyins,
                        beam_width=self.beam_width, device=self.device, dtype=beam_dtype,
                    )
                beam_acc.update_batch(beam_ids, targets)
                progress.update(beam_task, advance=len(chunk))
        progress.remove_task(beam_task)

        progress.remove_task(val_task)

        model.train()

        totals = torch.tensor(
            [
                val_cond_loss_sum,
                val_cond_tokens,
                val_uncond_loss_sum,
                val_uncond_tokens,
                beam_acc.correct_sentences,
                beam_acc.total_sentences,
            ],
            dtype=torch.float64,
            device=self.device,
        )
        if self.distributed.is_distributed:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)

        avg_cond_loss = (totals[0] / totals[1].clamp_min(1)).item()
        avg_uncond_loss = (totals[2] / totals[3].clamp_min(1)).item()
        avg_val_loss = avg_cond_loss + self.unconditional_loss_lambda * avg_uncond_loss
        topk_s_acc = (totals[4] / totals[5].clamp_min(1)).item()

        if self.distributed.is_distributed:
            gathered_states = [None] * self.distributed.world_size if self.is_main else None
            dist.gather_object(
                metrics_acc.state_dict(),
                object_gather_list=gathered_states,
                dst=0,
                group=self.distributed.object_group,
            )
            if self.is_main:
                metrics_acc.reset()
                for state in gathered_states:
                    metrics_acc.merge_state_dict(state)
                m = metrics_acc.compute()
            else:
                m = {}
        else:
            m = metrics_acc.compute()

        if self.log_cfg.log_with_wandb and self.is_main:
            log_data = {
                "val/loss": avg_val_loss,
                "val/conditional_loss": avg_cond_loss,
                "val/ACC": m["acc"],
                "val/Top3-ACC": m["top3_acc"],
                "val/Top5-ACC": m["top5_acc"],
                "val/S-ACC": m["s_acc"],
                f"val/S-ACC@{self.beam_width}-beam": topk_s_acc,
                "val/ECE": m["ece"],
            }
            if self.unconditional_loss_lambda != 0.0:
                log_data["val/unconditional_loss"] = avg_uncond_loss
            wandb.log(log_data, step=global_step)

        return avg_val_loss, m

    def train(self):
        # Validation intentionally bypasses the DDP wrapper: rank shards can
        # contain different batch counts, and inference needs no gradient sync.
        model = self.model.module if self.distributed.is_distributed else self.model
        cfg = self.cfg

        progress = _build_progress(cfg, disable=not self.is_main)
        global_step = 0

        with progress:
            epoch_task = progress.add_task("[yellow]Epochs", total=cfg.task.epochs, postfix="")
            decay_start_step = max(0, self.total_steps - cfg.task.decay_steps)
            before_decay_saved = False

            for epoch in range(cfg.task.epochs):

                batch_task = progress.add_task(
                    f"[green]Training Epoch {epoch + 1}/{self.cfg.task.epochs}",
                    total=self.epoch_steps,
                    postfix="[red]loss: N/A"
                )

                for batch in self.train_loader:
                    if not before_decay_saved and global_step == decay_start_step:
                        self._save_checkpoint(os.path.join(self.checkpoint_dir, "before_decay"))
                        before_decay_saved = True

                    self.optim.zero_grad()

                    with torch.amp.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                        out, flat_target_ids, _, flat_uncond_target = self._forward_batch(batch)

                    # Each rank's CE/focal loss is a mean over its local tokens.
                    # NJT batches have different token counts, so a plain DDP
                    # average would not equal the loss of the combined global
                    # batch. Scale each local mean such that DDP's gradient
                    # average becomes an exact global token-weighted mean.
                    local_token_counts = torch.tensor(
                        [
                            flat_target_ids.numel(),
                            int((flat_uncond_target != -100).sum().item()),
                        ],
                        dtype=torch.float64,
                        device=self.device,
                    )
                    global_token_counts = local_token_counts.clone()
                    if self.distributed.is_distributed:
                        dist.all_reduce(global_token_counts, op=dist.ReduceOp.SUM)

                    if self.distributed.is_distributed:
                        cond_scale = ddp_local_mean_scale(
                            local_token_counts[0],
                            global_token_counts[0],
                            self.distributed.world_size,
                        )
                        loss_for_backward = out.conditional_loss * cond_scale
                        if out.unconditional_loss is not None:
                            uncond_scale = ddp_local_mean_scale(
                                local_token_counts[1],
                                global_token_counts[1],
                                self.distributed.world_size,
                            )
                            loss_for_backward = loss_for_backward + (
                                self.unconditional_loss_lambda
                                * out.unconditional_loss
                                * uncond_scale
                            )
                    else:
                        loss_for_backward = out.loss

                    loss_for_backward.backward()

                    if cfg.task.gradient_clip_val > 0:
                        all_params = list(self.pre_model.parameters()) + list(self.post_model.parameters())
                        norm = torch.nn.utils.clip_grad_norm_(all_params, cfg.task.gradient_clip_val)
                    else:
                        norm = torch.tensor(0.0, device=self.device)

                    self.optim.step()
                    self.schd.step()

                    global_step += 1
                    train_stats = torch.stack([
                        out.conditional_loss.detach().double() * local_token_counts[0],
                        (
                            out.unconditional_loss.detach().double()
                            * local_token_counts[1]
                            if out.unconditional_loss is not None
                            else local_token_counts.new_zeros(())
                        ),
                        norm.detach().double(),
                    ])
                    if self.distributed.is_distributed:
                        dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
                    conditional_loss = (
                        train_stats[0] / global_token_counts[0].clamp_min(1)
                    ).item()
                    unconditional_loss = (
                        train_stats[1] / global_token_counts[1].clamp_min(1)
                    ).item()
                    scalar_loss = conditional_loss + (
                        self.unconditional_loss_lambda * unconditional_loss
                    )
                    grad_norm = (
                        train_stats[2] / self.distributed.world_size
                    ).item()

                    progress.update(batch_task, advance=1, postfix=f"[red]loss: {scalar_loss:.4f}")

                    if self.log_cfg.log_with_wandb and self.is_main:
                        log_data = {
                            "train/loss": scalar_loss,
                            "train/conditional_loss": conditional_loss,
                            "train/adamw_lr": self.optim.param_groups[0]["lr"],
                            "train/grad_norm": grad_norm,
                        }
                        if self.unconditional_loss_lambda != 0.0:
                            log_data["train/unconditional_loss"] = unconditional_loss
                        wandb.log(log_data, step=global_step)

                    # Validation
                    should_validate = (
                        global_step % self.log_cfg.val_interval == 0
                        or global_step == self.total_steps
                    )
                    if should_validate:
                        avg_val_loss, _metrics = self.validate(
                            model, self.val_loader, epoch, global_step, progress
                        )

                        progress.update(
                            epoch_task,
                            postfix=f"[red]val_loss={avg_val_loss:.4f}"
                        )

                    if global_step >= self.total_steps:
                        break

                progress.update(
                    epoch_task,
                    advance=1
                )

                progress.remove_task(batch_task)

                # Pre and post models are saved individually
                if (epoch + 1) % self.log_cfg.save_interval == 0:
                    save_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch + 1}")
                    self._save_checkpoint(save_path)

                if global_step >= self.total_steps:
                    break

        save_path = os.path.join(self.checkpoint_dir, "final_model")
        self._save_checkpoint(save_path)

        if self.log_cfg.log_with_wandb and self.is_main:
            wandb.finish()
