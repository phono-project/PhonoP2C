"""
PostfixLM training loop.

Trains both PhonoP2CPreModel and PhonoP2CPostModel jointly using NJT.
"""

import math
import os
import logging

import torch
import torch.nn as nn
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

from dataset import (
    make_collate_fn,
    create_dataset,
    transform_pinyin_predict_train,
    transform_pinyin_predict_val,
    P2CStreamingDataset
)
from datasets import load_from_disk, concatenate_datasets

from model.config import PreModelConfig, PostModelConfig, build_configs_from_dict
from model.model import PhonoP2CPreModel, PhonoP2CPostModel
from tokenizer import P2CTokenizer

from torchao.float8 import Float8LinearConfig, convert_to_float8_training
from utils.float8 import module_filter_fn
from loss import get_loss_fn
from metrics import MetricsAccumulator

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

        # Compile
        if cfg.task.compile_model:
            mode = cfg.task.get("compile_mode", "default")
            import torch._dynamo.config as dynamo_config
            dynamo_config.capture_scalar_outputs = True
            dynamo_config.capture_dynamic_output_shape_ops = True
            self.pre_model = torch.compile(self.pre_model, mode=mode, dynamic=True)
            self.post_model = torch.compile(self.post_model, mode=mode, dynamic=True)

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
            wandb.watch(self.pre_model, log="all", log_freq=cfg.logging.train.histogram_interval)
            wandb.watch(self.post_model, log="all", log_freq=cfg.logging.train.histogram_interval)

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
        self.loss_type = cfg.task.get("loss_type", "ce")

        # Loss function
        self.loss_fn = get_loss_fn(
            loss_type=self.loss_type,
            focal_loss_alpha=cfg.task.get("focal_loss_alpha", 0.25),
            focal_loss_gamma=cfg.task.get("focal_loss_gamma", 1.0),
            label_smoothing_epsilon=cfg.task.get("label_smoothing_epsilon", None),
        )
        self.use_amp = cfg.system.mixed_precision == "bf16"

    def validate(self, pre_model, post_model, loader, epoch, global_step, progress):
        pre_model.eval()
        post_model.eval()
        metrics_acc = MetricsAccumulator(ece_bins=self.cfg.task.ece_bins, ece_top_k=self.cfg.task.ece_top_k)
        val_steps = 0
        val_loss_sum = 0.0

        val_task = progress.add_task(f"[cyan]Validating Epoch {epoch + 1}/{self.cfg.task.epochs}", total=len(loader), postfix="")

        with torch.no_grad():
            for batch in loader:
                prefix_njt  = batch["prefix_ids_njt"].to(self.device)
                postfix_njt = batch["postfix_ids_njt"].to(self.device)
                target_njt  = batch["target_ids_njt"].to(self.device)

                flat_prefix_ids  = prefix_njt.values()
                prefix_offsets   = prefix_njt.offsets()
                flat_postfix_ids = postfix_njt.values()
                postfix_offsets  = postfix_njt.offsets()
                flat_target_ids  = target_njt.values()
                target_offsets   = target_njt.offsets()

                prefix_seq_lens = prefix_offsets[1:] - prefix_offsets[:-1]
                min_sl_pre = prefix_seq_lens.min().item()
                max_sl_pre = prefix_seq_lens.max().item()

                postfix_seq_lens = postfix_offsets[1:] - postfix_offsets[:-1]
                min_sl_post = postfix_seq_lens.min().item()
                max_sl_post = postfix_seq_lens.max().item()

                with torch.amp.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                    pre_embed, pre_K, pre_V = pre_model(
                        flat_prefix_ids, offsets=prefix_offsets,
                        min_seqlen=min_sl_pre, max_seqlen=max_sl_pre,
                    )
                    logits_njt = post_model(
                        flat_postfix_ids, input_offsets=postfix_offsets,
                        pre_K=pre_K, pre_V=pre_V, pre_offsets=prefix_offsets,
                        min_seqlen=min_sl_post, max_seqlen=max_sl_post,
                        min_seqlen_pre=min_sl_pre, max_seqlen_pre=max_sl_pre,
                    )
                    flat_logits = logits_njt.values()

                    loss = self.loss_fn(flat_logits, flat_target_ids)

                val_loss_sum += loss.item()
                val_steps += 1

                # Update metrics
                metrics_acc.update(flat_logits.detach(), flat_target_ids, target_offsets)

                progress.update(val_task, advance=1, postfix=f"[red]loss: {loss.item():.4f}")

        progress.remove_task(val_task)

        pre_model.train()
        post_model.train()

        avg_val_loss = val_loss_sum / max(val_steps, 1)
        m = metrics_acc.compute()
        avg_val_ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else float('inf')

        if self.log_cfg.log_with_wandb:
            wandb.log({
                "val/loss": avg_val_loss,
                "val/ACC": m["acc"],
                "val/Top3-ACC": m["top3_acc"],
                "val/Top5-ACC": m["top5_acc"],
                "val/S-ACC": m["s_acc"],
                "val/ECE": m["ece"],
            }, step=global_step)
            
        if self.loss_type == "ce":
            wandb.log({
                "val/PPL": avg_val_ppl
            }, step=global_step)

        return avg_val_loss, avg_val_ppl, m

    def train(self):
        pre_model = self.pre_model
        post_model = self.post_model
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
                    
                    prefix_njt  = batch["prefix_ids_njt"].to(self.device, non_blocking=True)
                    postfix_njt = batch["postfix_ids_njt"].to(self.device, non_blocking=True)
                    target_njt  = batch["target_ids_njt"].to(self.device, non_blocking=True)

                    flat_prefix_ids  = prefix_njt.values()
                    prefix_offsets   = prefix_njt.offsets()
                    flat_postfix_ids = postfix_njt.values()
                    postfix_offsets  = postfix_njt.offsets()
                    flat_target_ids  = target_njt.values()

                    prefix_seq_lens = prefix_offsets[1:] - prefix_offsets[:-1]
                    min_sl_pre = prefix_seq_lens.min().item()
                    max_sl_pre = prefix_seq_lens.max().item()

                    postfix_seq_lens = postfix_offsets[1:] - postfix_offsets[:-1]
                    min_sl_post = postfix_seq_lens.min().item()
                    max_sl_post = postfix_seq_lens.max().item()

                    # Zero grad
                    self.optim.zero_grad()

                    with torch.amp.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                        # Pre model: encode prefix
                        pre_embed, pre_K, pre_V = pre_model(
                            flat_prefix_ids, offsets=prefix_offsets,
                            min_seqlen=min_sl_pre, max_seqlen=max_sl_pre,
                        )

                        # Post model: decode pinyin -> Chinese
                        logits_njt = post_model(
                            flat_postfix_ids, input_offsets=postfix_offsets,
                            pre_K=pre_K, pre_V=pre_V, pre_offsets=prefix_offsets,
                            min_seqlen=min_sl_post, max_seqlen=max_sl_post,
                            min_seqlen_pre=min_sl_pre, max_seqlen_pre=max_sl_pre,
                        )

                        flat_logits = logits_njt.values()

                        loss = self.loss_fn(flat_logits, flat_target_ids)

                    loss.backward()

                    if cfg.task.gradient_clip_val > 0:
                        all_params = list(pre_model.parameters()) + list(post_model.parameters())
                        norm = torch.nn.utils.clip_grad_norm_(all_params, cfg.task.gradient_clip_val)

                    self.optim.step()
                    self.schd.step()

                    global_step += 1
                    scalar_loss = loss.item()

                    progress.update(batch_task, advance=1, postfix=f"[red]loss: {scalar_loss:.4f}")

                    if self.log_cfg.log_with_wandb:
                        log_data = {
                            "train/loss": scalar_loss,
                            "train/adamw_lr": self.optim.param_groups[0]["lr"],
                            "train/grad_norm": norm.item()
                        }
                        wandb.log(log_data, step=global_step)
                        
                    # Validation
                    if (global_step % self.log_cfg.val_interval == 0) or (global_step == self.total_steps):
                        avg_val_loss, avg_val_ppl, _metrics = self.validate(
                            pre_model, post_model, self.val_loader, epoch, global_step, progress
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

                # Save checkpoint
                if (epoch + 1) % self.log_cfg.save_interval == 0:
                    save_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch + 1}")

                    pre_to_save = pre_model._orig_mod if hasattr(pre_model, "_orig_mod") else pre_model
                    post_to_save = post_model._orig_mod if hasattr(post_model, "_orig_mod") else post_model

                    pre_to_save.save_pretrained(os.path.join(save_path, "pre_model"), safe_serialization=True)
                    post_to_save.save_pretrained(os.path.join(save_path, "post_model"), safe_serialization=True)

        save_path = os.path.join(self.checkpoint_dir, f"final_model")
        
        pre_to_save = pre_model._orig_mod if hasattr(pre_model, "_orig_mod") else pre_model
        post_to_save = post_model._orig_mod if hasattr(post_model, "_orig_mod") else post_model
        
        pre_to_save.save_pretrained(os.path.join(save_path, "pre_model"), safe_serialization=True)
        post_to_save.save_pretrained(os.path.join(save_path, "post_model"), safe_serialization=True)
        
        if self.log_cfg.log_with_wandb:
            wandb.finish()