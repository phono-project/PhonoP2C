"""
PostfixLM training loop (new-standard encoder-decoder architecture).

Trains both PhonoP2CPreModel (causal decoder) and PhonoP2CPostModel (pinyin
encoder) jointly using NJT batches.  The whole two-pass forward (unconditional
pass + conditional cross-attended pass) is wrapped in PhonoP2CTrainWrapper and
torch.compile is applied to the wrapper.  The two losses (unconditional /
conditional) are logged separately; pre and post models are saved
individually.
"""

import math
import os
import logging

import torch
import bitsandbytes as bnb
from transformers import get_scheduler
import wandb
from streaming import StreamingDataLoader
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
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


def _build_progress(cfg: DictConfig) -> Progress:
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
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.system.device)

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
        print("Building pinyin->Chinese possibility map...")
        logits_mask = self.tokenizer.create_possibility_map().to(self.device)
        print(f"  Mask: {logits_mask.shape}, nonzero={logits_mask.sum().item()} "
              f"({100 * logits_mask.sum().item() / logits_mask.numel():.1f}%)")

        # Build models
        self.pre_model = PhonoP2CPreModel(self.pre_cfg).to(self.device)

        self.post_model = PhonoP2CPostModel(self.post_cfg).to(self.device)
        self.post_model.logits_mask = logits_mask

        total_params = (sum(p.numel() for p in self.pre_model.parameters()) +
                        sum(p.numel() for p in self.post_model.parameters()))
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
        if cfg.system.get("gradient_checkpointing", False):
            self.pre_model.gradient_checkpointing = True
            self.post_model.gradient_checkpointing = True

        # Loss function (delivered into the wrapper so the loss is computed
        # inside the compiled region for better optimization).
        self.loss_type = cfg.task.get("loss_type", "ce")
        self.loss_fn = get_loss_fn(
            loss_type=self.loss_type,
            focal_loss_alpha=cfg.task.get("focal_loss_alpha", 0.25),
            focal_loss_gamma=cfg.task.get("focal_loss_gamma", 1.0),
            label_smoothing_epsilon=cfg.task.get("label_smoothing_epsilon", None),
        )

        # Two-pass training wrapper
        self.model = PhonoP2CTrainWrapper(self.pre_model, self.post_model, self.loss_fn)

        # Compile the wrapped forward pass
        if cfg.task.compile_model:
            mode = cfg.task.get("compile_mode", "default")
            import torch._dynamo.config as dynamo_config
            dynamo_config.capture_scalar_outputs = True
            dynamo_config.capture_dynamic_output_shape_ops = True
            self.model = torch.compile(self.model, mode=mode, dynamic=True)

        # Logging
        self.log_cfg = cfg.logging.train
        self.checkpoint_dir = self.log_cfg.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.log_cfg.log_with_wandb:
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
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=cfg.task.batchsize,
            shuffle=False,
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
                    self.optim = bnb.optim.PagedAdamW8bit(param_groups, lr=cfg.task.learning_rate, weight_decay=cfg.task.weight_decay)
                else:
                    self.optim = bnb.optim.AdamW8bit(param_groups, lr=cfg.task.learning_rate, weight_decay=cfg.task.weight_decay)
            else:
                if cfg.system.optim_paged:
                    self.optim = bnb.optim.PagedAdamW32bit(param_groups, lr=cfg.task.learning_rate, weight_decay=cfg.task.weight_decay)
                else:
                    self.optim = bnb.optim.AdamW32bit(param_groups, lr=cfg.task.learning_rate, weight_decay=cfg.task.weight_decay)
            self.schd = get_scheduler(
                "cosine", self.optim, num_warmup_steps=warmup_steps, num_training_steps=self.total_steps
            )
        elif cfg.task.optimizer == "ademamix":
            if cfg.system.optim_8bit:
                if cfg.system.optim_paged:
                    self.optim = bnb.optim.PagedAdEMAMix8bit(param_groups, lr=cfg.task.learning_rate, weight_decay=cfg.task.weight_decay)
                else:
                    self.optim = bnb.optim.PagedAdEMAMix8bit(param_groups, lr=cfg.task.learning_rate, weight_decay=cfg.task.weight_decay)
            else:
                if cfg.system.optim_paged:
                    self.optim = bnb.optim.PagedAdEMAMix32bit(param_groups, lr=cfg.task.learning_rate, weight_decay=cfg.task.weight_decay)
                else:
                    self.optim = bnb.optim.PagedAdEMAMix32bit(param_groups, lr=cfg.task.learning_rate, weight_decay=cfg.task.weight_decay)
            self.schd = get_scheduler(
                "cosine", self.optim, num_warmup_steps=warmup_steps, num_training_steps=self.total_steps
            )
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

    def _forward_batch(self, batch):
        full_prefix_njt = batch["full_prefix_ids_njt"].to(self.device)
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

        out = self.model(
            flat_prefix, prefix_offsets, flat_suffix, suffix_offsets,
            flat_postfix, flat_uncond_target, flat_target_ids, prefix_lens,
            min_sl_prefix, max_sl_prefix, min_sl_suffix, max_sl_suffix,
            min_sl_full, max_sl_full,
        )

        return out, flat_target_ids, suffix_offsets, full_prefix_njt, postfix_njt, target_njt

    def validate(self, model, loader, epoch, global_step, progress):
        model.eval()
        metrics_acc = MetricsAccumulator(ece_bins=self.cfg.task.ece_bins, ece_top_k=self.cfg.task.ece_top_k)
        beam_acc = TopKSentenceAccuracy(k=self.beam_width)
        val_steps = 0
        val_loss_sum = 0.0
        val_uncond_loss_sum = 0.0
        val_cond_loss_sum = 0.0

        pre_model = self.pre_model
        post_model = self.post_model

        # Collect all beam-search samples in dataset order (so strided
        # sampling spreads evenly across sources), then group by pinyin length
        # so each group can be decoded with a large, efficient batch.
        beam_samples: list[tuple] = []

        val_task = progress.add_task(f"[cyan]Validating Epoch {epoch + 1}/{self.cfg.task.epochs}", total=len(loader), postfix="")

        with torch.no_grad():
            for batch in loader:
                with torch.amp.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                    out, flat_target_ids, target_offsets, full_prefix_njt, postfix_njt, target_njt = \
                        self._forward_batch(batch)

                val_loss_sum += out.loss.item()
                val_uncond_loss_sum += out.unconditional_loss.item()
                val_cond_loss_sum += out.conditional_loss.item()
                val_steps += 1

                # Update metrics (conditional logits are the predictions)
                metrics_acc.update(out.conditional_logits.detach(), flat_target_ids, target_offsets)

                # Collect beam-search samples.
                for prefix_t, postfix_t, target_t in zip(
                    full_prefix_njt.unbind(), postfix_njt.unbind(), target_njt.unbind()
                ):
                    beam_samples.append((prefix_t.tolist(), postfix_t.tolist(), target_t))

                progress.update(val_task, advance=1, postfix=f"[red]loss: {out.loss.item():.4f}")

        # Strided sampling (even, unbiased spread) then group by pinyin length.
        sampled = beam_samples[:: self.beam_stride]
        beam_groups: dict[int, list] = {}
        for prefix, pinyin, target in sampled:
            beam_groups.setdefault(len(pinyin), []).append((prefix, pinyin, target))

        # Standalone progress bar for the beam-search metric.
        beam_dtype = torch.bfloat16 if self.use_amp else None
        beam_task = progress.add_task(
            f"[cyan]S-ACC@{self.beam_width}-beam",
            total=len(sampled),
            postfix=f"[red]{len(sampled)} samples (1/{self.beam_stride})",
        )
        for T, group in beam_groups.items():
            for i in range(0, len(group), self.beam_chunk_size):
                chunk = group[i:i + self.beam_chunk_size]
                prefixes = [c[0] for c in chunk]
                pinyins = [c[1] for c in chunk]
                targets = torch.stack([c[2] for c in chunk])
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

        avg_val_loss = val_loss_sum / max(val_steps, 1)
        avg_uncond_loss = val_uncond_loss_sum / max(val_steps, 1)
        avg_cond_loss = val_cond_loss_sum / max(val_steps, 1)
        m = metrics_acc.compute()
        topk_s_acc = beam_acc.compute()

        if self.log_cfg.log_with_wandb:
            wandb.log({
                "val/loss": avg_val_loss,
                "val/unconditional_loss": avg_uncond_loss,
                "val/conditional_loss": avg_cond_loss,
                "val/ACC": m["acc"],
                "val/Top3-ACC": m["top3_acc"],
                "val/Top5-ACC": m["top5_acc"],
                "val/S-ACC": m["s_acc"],
                f"val/S-ACC@{self.beam_width}-beam": topk_s_acc,
                "val/ECE": m["ece"],
            }, step=global_step)

        return avg_val_loss, m

    def train(self):
        model = self.model
        cfg = self.cfg

        progress = _build_progress(cfg)
        global_step = 0

        with progress:
            epoch_task = progress.add_task("[yellow]Epochs", total=cfg.task.epochs, postfix="")

            for epoch in range(cfg.task.epochs):

                batch_task = progress.add_task(
                    f"[green]Training Epoch {epoch + 1}/{self.cfg.task.epochs}",
                    total=self.epoch_steps,
                    postfix="[red]loss: N/A"
                )

                for batch in self.train_loader:

                    self.optim.zero_grad()

                    with torch.amp.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                        out, *_ = self._forward_batch(batch)

                    out.loss.backward()

                    if cfg.task.gradient_clip_val > 0:
                        all_params = list(self.pre_model.parameters()) + list(self.post_model.parameters())
                        norm = torch.nn.utils.clip_grad_norm_(all_params, cfg.task.gradient_clip_val)
                    else:
                        norm = torch.tensor(0.0)

                    self.optim.step()
                    self.schd.step()

                    global_step += 1
                    scalar_loss = out.loss.item()

                    progress.update(batch_task, advance=1, postfix=f"[red]loss: {scalar_loss:.4f}")

                    if self.log_cfg.log_with_wandb:
                        log_data = {
                            "train/loss": scalar_loss,
                            "train/unconditional_loss": out.unconditional_loss.item(),
                            "train/conditional_loss": out.conditional_loss.item(),
                            "train/adamw_lr": self.optim.param_groups[0]["lr"],
                            "train/grad_norm": norm.item()
                        }
                        wandb.log(log_data, step=global_step)

                    # Validation
                    if (global_step % self.log_cfg.val_interval == 0) or (global_step == self.total_steps):
                        avg_val_loss, _metrics = self.validate(
                            model, self.val_loader, epoch, global_step, progress
                        )

                        progress.update(
                            epoch_task,
                            postfix=f"[red]val_loss={avg_val_loss:.4f}"
                        )

                progress.update(
                    epoch_task,
                    advance=1
                )

                progress.remove_task(batch_task)

                # Pre and post models are saved individually
                if (epoch + 1) % self.log_cfg.save_interval == 0:
                    save_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch + 1}")

                    pre_to_save = self.pre_model._orig_mod if hasattr(self.pre_model, "_orig_mod") else self.pre_model
                    post_to_save = self.post_model._orig_mod if hasattr(self.post_model, "_orig_mod") else self.post_model

                    pre_to_save.save_pretrained(os.path.join(save_path, "pre_model"), safe_serialization=True)
                    post_to_save.save_pretrained(os.path.join(save_path, "post_model"), safe_serialization=True)

        save_path = os.path.join(self.checkpoint_dir, "final_model")

        pre_to_save = self.pre_model._orig_mod if hasattr(self.pre_model, "_orig_mod") else self.pre_model
        post_to_save = self.post_model._orig_mod if hasattr(self.post_model, "_orig_mod") else self.post_model

        pre_to_save.save_pretrained(os.path.join(save_path, "pre_model"), safe_serialization=True)
        post_to_save.save_pretrained(os.path.join(save_path, "post_model"), safe_serialization=True)

        if self.log_cfg.log_with_wandb:
            wandb.finish()
