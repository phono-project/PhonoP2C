"""
Dictionary N-Best hyperparameter search with Optuna.
 
Subtask 1 — Dictionary preprocessing & probs extraction:
  1. Load & filter dictionary (chinese-vocab check, multi-char only, dedup)
  2. Build trie and save dict_trie.json
  3. Run model inference on validation set, extract per-position non-zero probs
     sorted by probability, convert token IDs -> Chinese chars, save probs.json
 
Subtask 2 — Optuna hyperparameter search:
  1. Load probs.json & dict_trie.json
  2. Precompute trie word-matches per sample position
  3. Viterbi beam-search DP: score = sum(ln(probs)) + n_words * ln(beta)
  4. Optimize beta via Optuna, report Sentence-ACC & N-Sentence-ACC
  5. Save results.json
"""

import json
import math
import os

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from algo.trie import load_dictionary, build_trie, find_matching_words
from algo.viterbi_dp import viterbi_nbest
from model.model import PhonoP2CPreModel, PhonoP2CPostModel
from model.utils import gather_target_logits
from tokenizer import P2CTokenizer
from datasets_pipeline import make_collate_fn, create_dataset, transform_pinyin_predict_val


import concurrent.futures
from typing import Any

# Global variable to store Trie in worker processes to avoid IPC overhead
_global_trie: dict | None = None


def init_worker_trie(trie: dict) -> None:
    """Initializer function for worker processes to load the shared Trie."""
    global _global_trie
    _global_trie = trie


def process_json_lines_chunk(lines: list[str]) -> list[tuple[list[dict[str, float]], list[list[tuple[int, str, float]]], str]]:
    """Process a chunk of raw JSON lines in a worker process."""
    global _global_trie
    if _global_trie is None:
        raise ValueError("Trie has not been initialized in the worker process.")

    results = []
    for line in lines:
        smp = json.loads(line)
        candidates: list[dict[str, float]] = []
        for pos_cands in smp["positions"]:
            d: dict[str, float] = {}
            for item in pos_cands:
                d[item["c"]] = float(item["p"])
            candidates.append(d)

        # Precompute matching words using the globally shared Trie
        words_at: list[list[tuple[int, str, float]]] = [
            find_matching_words(_global_trie, candidates, i)
            for i in range(len(candidates))
        ]
        results.append((candidates, words_at, smp["target"]))
    return results


# Parallel Viterbi evaluation
def evaluate_chunk_viterbi(
    chunk: list[tuple[list[dict[str, float]], list[list[tuple[int, str, float]]], str]],
    beta_single: float,
    beta_word: float,
    N: int,
) -> tuple[int, int, int]:
    """Run Viterbi beam search on a precomputed data chunk."""
    correct_1 = 0
    correct_N = 0
    total = 0

    for candidates, words_at, target in chunk:
        best = viterbi_nbest(candidates, words_at, beta_single, beta_word, N)
        total += 1
        if not best:
            continue
        
        joined = "".join(best[0][1])
        if joined == target:
            correct_1 += 1
            correct_N += 1
        else:
            for _, wlist in best[1:]:
                if "".join(wlist) == target:
                    correct_N += 1
                    break

    return correct_1, correct_N, total


# Main runner
class ParamSearchRunner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.system.device)
        cfg_t = cfg.task
        os.makedirs(cfg_t.output_dir, exist_ok=True)
        self.tokenizer = P2CTokenizer.from_config(cfg_t.vocabs_config)

        self._trie: dict | None = None
        self._precomputed: list[tuple[list[dict[str, float]],
                                      list[list[tuple[int, str, float]]],
                                      str]] = []

    def run(self) -> None:
        subtasks = self.cfg.task.subtasks
        if subtasks.get("preprocess", True):
            self._subtask_preprocess()
        if subtasks.get("optuna_search", True):
            self._subtask_optuna_search()

    # Subtask 1
    def _subtask_preprocess(self) -> None:
        cfg = self.cfg.task
        tokenizer = self.tokenizer

        # Load & filter dictionary
        print("Loading dictionary …")
        raw = load_dictionary(cfg.dict_path)
        print(f"  Raw entries: {len(raw)}")

        filtered: list[str] = []
        for w in raw:
            if not tokenizer.check(w, vocab="chinese"):
                continue
            if len(w) <= 1:
                continue
            if len(w) >= cfg.dict.max_len:
                continue
            filtered.append(w)
        unique = list(dict.fromkeys(filtered))  # dedup preserving order
        print(f"  After filtering + dedup: {len(unique)}")

        # Build & save trie
        trie = build_trie(unique)
        trie_path = os.path.join(cfg.output_dir, "dict_trie.json")
        with open(trie_path, "w", encoding="utf-8") as f:
            json.dump(trie, f, ensure_ascii=False)
        print(f"Trie saved -> {trie_path}")

        # Load pretrained model
        print("Loading pretrained model …")
        pre_model = PhonoP2CPreModel.from_pretrained(
            cfg.pretrained_pre_model
        ).to(self.device)
        post_model = PhonoP2CPostModel.from_pretrained(
            cfg.pretrained_post_model
        ).to(self.device)

        logits_mask = tokenizer.create_possibility_map().to(self.device)
        post_model.logits_mask = logits_mask

        pre_model.eval()
        post_model.eval()

        # Validation dataset
        val_ds = create_dataset(cfg.val_dir, keep_in_memory=True)
        val_ds = val_ds.shuffle()
        val_ds = val_ds.train_test_split(train_size=cfg.num_samples)["train"]
        
        val_ds.set_transform(
            lambda batch: transform_pinyin_predict_val(batch, tokenizer, None)
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batchsize,
            shuffle=False,
            collate_fn=make_collate_fn(),
            num_workers=0,
        )

        use_amp = self.cfg.system.mixed_precision == "bf16"

        # Inference & probs extraction
        print("Running inference …")
        probs_path = os.path.join(cfg.output_dir, "probs.json")
        sample_count = 0

        with open(probs_path, "w", encoding="utf-8") as f_out, torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                pre_njt = batch["pre_ids_njt"].to(self.device)
                postfix_njt = batch["postfix_ids_njt"].to(self.device)
                target_njt = batch["target_ids_njt"].to(self.device)

                flat_pre_ids = pre_njt.values()
                pre_offsets = pre_njt.offsets()
                flat_postfix_ids = postfix_njt.values()
                postfix_offsets = postfix_njt.offsets()

                pre_seq_lens = pre_offsets[1:] - pre_offsets[:-1]
                min_sl_pre = int(pre_seq_lens.min().item())
                max_sl_pre = int(pre_seq_lens.max().item())

                postfix_seq_lens = postfix_offsets[1:] - postfix_offsets[:-1]
                min_sl_post = int(postfix_seq_lens.min().item())
                max_sl_post = int(postfix_seq_lens.max().item())

                with torch.amp.autocast(self.device.type, dtype=torch.bfloat16, enabled=use_amp):
                    _, past_kv = pre_model(
                        flat_pre_ids, offsets=pre_offsets,
                        min_seqlen=min_sl_pre, max_seqlen=max_sl_pre,
                    )
                    post_hidden, post_mask = post_model(
                        flat_postfix_ids, input_offsets=postfix_offsets,
                        min_seqlen=min_sl_post, max_seqlen=max_sl_post,
                    )
                    logits_cond_njt, _ = pre_model(
                        flat_pre_ids, offsets=pre_offsets,
                        min_seqlen=min_sl_pre, max_seqlen=max_sl_pre,
                        past_kv=past_kv,
                        post_hidden=post_hidden, post_offsets=postfix_offsets,
                        min_seqlen_post=min_sl_post, max_seqlen_post=max_sl_post,
                        logits_mask=post_mask,
                    )

                # Target-aligned conditional logits
                flat_cond = gather_target_logits(
                    logits_cond_njt.values(), pre_offsets, postfix_offsets
                )
                cond_njt = torch.nested.nested_tensor_from_jagged(
                    flat_cond, postfix_offsets, min_seqlen=min_sl_post, max_seqlen=max_sl_post
                )

                # Split NJT into per-sample tensors
                for sample_logits, sample_target in zip(
                    cond_njt.unbind(), target_njt.unbind()
                ):
                    l_cpu = sample_logits.cpu()
                    t_cpu = sample_target.cpu()
                    probs = l_cpu.softmax(dim=-1)
                    target_text = tokenizer.ids_to_text(t_cpu.tolist())

                    positions: list[list[dict]] = []
                    for pos in range(probs.shape[0]):
                        p = probs[pos]
                        threshold = cfg.epsilon
                        nonzero = p > threshold

                        ids = torch.where(nonzero)[0]
                        vals = p[nonzero]
                        ids = torch.where(nonzero)[0]
                        vals = p[nonzero]
                        if ids.numel() == 0:
                            positions.append([])
                            continue
                        order = torch.argsort(vals, descending=True)
                        sorted_ids = ids[order].tolist()
                        sorted_probs = vals[order].tolist()

                        chars = [
                            tokenizer._id_to_chinese.get(tid, "")
                            for tid in sorted_ids
                        ]
                        positions.append([
                            {"c": ch, "p": float(pr)} for ch, pr in zip(chars, sorted_probs)
                        ])

                    f_out.write(json.dumps(
                        {"target": target_text, "positions": positions},
                        ensure_ascii=False,
                    ) + "\n")
                    sample_count += 1

                if (batch_idx + 1) % 10 == 0:
                    print(f"  processed {batch_idx + 1}/{len(val_loader)} batches, "
                          f"{sample_count} samples so far")

        print(f"Probs saved -> {probs_path}  ({sample_count} samples)")

    # Subtask 2
    def _subtask_optuna_search(self) -> None:
        cfg = self.cfg.task
        opt_cfg = cfg.optuna

        probs_path = os.path.join(cfg.output_dir, "probs.json")
        trie_path = os.path.join(cfg.output_dir, "dict_trie.json")

        print("Loading dict_trie.json …")
        with open(trie_path, "r", encoding="utf-8") as f:
            trie = json.load(f)
        self._trie = trie

        # Parallel Precomputation
        print("Loading & precomputing word matches in parallel …")
        self._precomputed = []
        
        # Read all lines from JSON
        with open(probs_path, "r", encoding="utf-8") as f:
            all_lines = f.readlines()

        print(f"  Processing subset of {len(all_lines)} samples")

        # Determine optimal chunk size for worker processes
        num_workers = os.cpu_count() or 4
        chunk_size = max(1, len(all_lines) // (num_workers * 4))
        line_chunks = [all_lines[i:i + chunk_size] for i in range(0, len(all_lines), chunk_size)]

        # Run process pool to precompute data chunks
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker_trie,
            initargs=(self._trie,)
        ) as executor:
            futures = [executor.submit(process_json_lines_chunk, chunk) for chunk in line_chunks]
            
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Precomputing"):
                self._precomputed.extend(fut.result())
                
        print(f"  done. ({len(self._precomputed)} samples successfully loaded and precomputed)")

        # Split precomputed dataset into fixed chunks for parallelized Optuna trials
        self.num_eval_workers = num_workers
        eval_chunk_size = max(1, len(self._precomputed) // self.num_eval_workers)
        self._eval_chunks = [
            self._precomputed[i:i + eval_chunk_size] 
            for i in range(0, len(self._precomputed), eval_chunk_size)
        ]

        # Optuna study
        import optuna

        objective_fn = (
            self._objective_s_acc
            if opt_cfg.get("objective", "sentence_acc") == "sentence_acc"
            else self._objective_n_s_acc
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.GPSampler(seed=42),
        )
        study.optimize(
            objective_fn,
            n_trials=opt_cfg.n_trials,
            show_progress_bar=True,
        )

        best_trial = study.best_trial
        best_beta_single = best_trial.params["beta_single"]
        best_beta_word = best_trial.params["beta_word"]
        best_s_acc = best_trial.user_attrs.get("sentence_acc", 0.0)
        best_n_s_acc = best_trial.user_attrs.get("n_sentence_acc", 0.0)

        results = {
            "best_beta_single": best_beta_single,
            "best_beta_word": best_beta_word,
            "sentence_acc": best_s_acc,
            "n_sentence_acc": best_n_s_acc,
            "n_best": opt_cfg.n_best,
            "n_trials": opt_cfg.n_trials,
        }

        print("\n" + "=" * 60)
        print("OPTUNA HYPERPARAMETER SEARCH RESULTS")
        print("=" * 60)
        print(f"  Best beta_single     : {best_beta_single:.6e}")
        print(f"  Best beta_word       : {best_beta_word:.6e}")
        print(f"  Sentence-ACC         : {best_s_acc:.4f}  ({best_s_acc * 100:.2f}%)")
        print(f"  N-Sentence-ACC (N={results['n_best']}): {best_n_s_acc:.4f}  ({best_n_s_acc * 100:.2f}%)")
        print("=" * 60)

        results_path = os.path.join(cfg.output_dir, "results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved -> {results_path}")

    # Optuna objs
    def _objective_s_acc(self, trial: "optuna.Trial") -> float:
        opt_cfg = self.cfg.task.optuna
        
        # Suggest prior probabilities
        beta_single = trial.suggest_float(
            "beta_single", 
            opt_cfg.get("beta_single_low", 1e-4), 
            opt_cfg.get("beta_single_high", 0.5), 
            log=False
        )
        beta_word = trial.suggest_float(
            "beta_word", 
            opt_cfg.get("beta_word_low", 1e-2), 
            opt_cfg.get("beta_word_high", 1.0), 
            log=False
        )
        N = opt_cfg.n_best

        total_correct_1 = 0
        total_correct_N = 0
        total_samples = 0

        # Parallel evaluation across worker processes
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_eval_workers) as executor:
            futures = [
                executor.submit(evaluate_chunk_viterbi, chunk, beta_single, beta_word, N)
                for chunk in self._eval_chunks
            ]
            for fut in concurrent.futures.as_completed(futures):
                c1, cN, tot = fut.result()
                total_correct_1 += c1
                total_correct_N += cN
                total_samples += tot

        s_acc = total_correct_1 / total_samples if total_samples > 0 else 0.0
        n_s_acc = total_correct_N / total_samples if total_samples > 0 else 0.0
        
        trial.set_user_attr("sentence_acc", s_acc)
        trial.set_user_attr("n_sentence_acc", n_s_acc)
        return s_acc

    def _objective_n_s_acc(self, trial: "optuna.Trial") -> float:
        opt_cfg = self.cfg.task.optuna
        
        beta_single = trial.suggest_float(
            "beta_single", 
            opt_cfg.get("beta_single_low", 1e-4), 
            opt_cfg.get("beta_single_high", 0.5), 
            log=True
        )
        beta_word = trial.suggest_float(
            "beta_word", 
            opt_cfg.get("beta_word_low", 1e-2), 
            opt_cfg.get("beta_word_high", 1.0), 
            log=True
        )
        N = opt_cfg.n_best

        total_correct_1 = 0
        total_correct_N = 0
        total_samples = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_eval_workers) as executor:
            futures = [
                executor.submit(evaluate_chunk_viterbi, chunk, beta_single, beta_word, N)
                for chunk in self._eval_chunks
            ]
            for fut in concurrent.futures.as_completed(futures):
                c1, cN, tot = fut.result()
                total_correct_1 += c1
                total_correct_N += cN
                total_samples += tot

        s_acc = total_correct_1 / total_samples if total_samples > 0 else 0.0
        n_s_acc = total_correct_N / total_samples if total_samples > 0 else 0.0
        
        trial.set_user_attr("sentence_acc", s_acc)
        trial.set_user_attr("n_sentence_acc", n_s_acc)
        return n_s_acc