"""Shared train/eval loop used by every method (lora, full_ft, and later
adapters/prefix_tuning), so run_experiment.py has one call site regardless
of which method module produced the trainable model.

Console output is kept to a couple of continuously-updating progress bars
(one for the training steps, one for the generation-based eval) instead of
the library's default per-logging-step printouts — a 21-run task x rank
sweep would otherwise flood a Colab cell with thousands of lines. Full
detail (every logged step, library internals) still goes to
`<run_dir>/train.log` via utils.logging_utils.file_logging, so nothing is
actually lost.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from config.schema import ExperimentConfig, MethodSpec
from data.loaders import get_choices, tokenize_for_causal_lm
from eval.metrics import task_accuracy, write_metric
from utils.logging_utils import file_logging, get_logger

log = get_logger(__name__)

_BAR_FORMAT = "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}"


def method_run_name(method: MethodSpec) -> str:
    if method.type == "lora":
        return f"lora_r{method.rank}"
    return method.type


def count_trainable_parameters(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


class CompactProgressCallback:
    """A HF Trainer callback that shows one continuously-updating tqdm bar
    (with a live wall-clock timer via tqdm's {elapsed}) instead of printing
    a line per logging step. Every step's metrics still get written to
    `run_logger` (which, under utils.logging_utils.file_logging, lands only
    in the run's log file, not the console).
    """

    def __init__(self, desc: str, run_logger: logging.Logger):
        self.desc = desc
        self.run_logger = run_logger
        self.bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        from tqdm.auto import tqdm

        self.bar = tqdm(total=state.max_steps, desc=self.desc, unit="step", dynamic_ncols=True, bar_format=_BAR_FORMAT)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        self.run_logger.info("step=%s %s", state.global_step, logs)
        if self.bar is not None and "loss" in logs:
            self.bar.set_postfix_str(f"loss={logs['loss']:.4f}")

    def on_step_end(self, args, state, control, **kwargs):
        if self.bar is not None:
            self.bar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        if self.bar is not None:
            self.bar.close()


def generate_predictions(
    model,
    tokenizer,
    eval_dataset,
    max_new_tokens: int,
    max_seq_length: int,
    num_samples: int,
    desc: str = "eval",
) -> tuple[list[str], list[str]]:
    import torch
    from tqdm.auto import tqdm

    was_training = model.training
    model.eval()
    predictions: list[str] = []
    references: list[str] = []
    n = min(num_samples, len(eval_dataset))
    bar = tqdm(range(n), desc=desc, unit="ex", dynamic_ncols=True, bar_format=_BAR_FORMAT)
    with torch.no_grad():
        for i in bar:
            example = eval_dataset[i]
            inputs = tokenizer(
                example["prompt"],
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
            ).to(model.device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            generated = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            predictions.append(generated)
            references.append(example["completion"])
    bar.close()
    model.train(was_training)
    return predictions, references


def training_curve_from_trainer(trainer) -> list[dict[str, float]]:
    """Extract {step, loss} points from the Trainer's log history, for Plot #3."""
    return [
        {"step": entry["step"], "loss": entry["loss"]}
        for entry in trainer.state.log_history
        if "loss" in entry and "step" in entry
    ]


def train_and_evaluate(
    model,
    tokenizer,
    task: str,
    train_dataset,
    eval_dataset,
    method: MethodSpec,
    config: ExperimentConfig,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Train `model` (already wrapped for `method`) on `task` and score it.

    Writes one JSON-lines record to `output_dir/metrics.jsonl` per
    (task, method) run so experiment 1's task x rank sweep produces a
    single file scripts/plot_results.py can read for plotting, plus a full
    detail log to `output_dir/<run_name>/train.log`.
    """
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
    from transformers.trainer_callback import PrinterCallback, ProgressCallback

    training_cfg = config.training
    run_name = f"{task}__{method_run_name(method)}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    run_logger = logging.getLogger(f"lora_experiments.runs.{run_name}")

    with file_logging(log_path):
        run_logger.info("Starting run=%s task=%s method=%s", run_name, task, method.model_dump(exclude_none=True))
        start = time.time()

        tokenized_train = tokenize_for_causal_lm(train_dataset, tokenizer, training_cfg.max_seq_length)

        args = TrainingArguments(
            output_dir=str(run_dir),
            num_train_epochs=training_cfg.epochs,
            per_device_train_batch_size=training_cfg.per_device_train_batch_size,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            learning_rate=method.learning_rate or training_cfg.learning_rate,
            logging_steps=10,
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
            bf16=device == "cuda",
            seed=config.seed,
        )
        collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_train,
            data_collator=collator,
        )
        trainer.pop_callback(PrinterCallback)
        trainer.pop_callback(ProgressCallback)
        trainer.add_callback(CompactProgressCallback(f"[{run_name}] train", run_logger))

        train_result = trainer.train()

        choices = get_choices(task)
        predictions, references = generate_predictions(
            model,
            tokenizer,
            eval_dataset,
            max_new_tokens=training_cfg.max_new_tokens,
            max_seq_length=training_cfg.max_seq_length,
            num_samples=training_cfg.eval_samples,
            desc=f"[{run_name}] eval ",
        )
        accuracy = task_accuracy(predictions, references, choices)
        trainable_params, total_params = count_trainable_parameters(model)
        elapsed = time.time() - start

        metrics = {
            "experiment": config.experiment,
            "task": task,
            "run_name": run_name,
            "method": method.model_dump(exclude_none=True),
            "train_loss": train_result.training_loss,
            "eval_accuracy": accuracy,
            "eval_samples": len(references),
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_param_pct": trainable_params / total_params if total_params else 0.0,
            "elapsed_seconds": elapsed,
            "training_curve": training_curve_from_trainer(trainer),
        }
        write_metric(output_dir / "metrics.jsonl", metrics)
        run_logger.info(
            "Finished run=%s accuracy=%.4f trainable_params=%d/%d (%.4f%%) elapsed=%.1fs",
            run_name,
            accuracy,
            trainable_params,
            total_params,
            100 * metrics["trainable_param_pct"],
            elapsed,
        )

    print(
        f"[{run_name}] done in {elapsed:.0f}s — accuracy={accuracy:.4f}, "
        f"trainable={trainable_params:,}/{total_params:,} ({100 * metrics['trainable_param_pct']:.2f}%), "
        f"log={log_path}"
    )
    return metrics
