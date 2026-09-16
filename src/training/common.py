"""Shared train/eval loop used by every method (lora, full_ft, and later
adapters/prefix_tuning), so run_experiment.py has one call site regardless
of which method module produced the trainable model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from lora_experiments.config.schema import ExperimentConfig, MethodSpec
from lora_experiments.data.loaders import get_choices, tokenize_for_causal_lm
from lora_experiments.eval.metrics import task_accuracy, write_metric
from lora_experiments.utils.logging_utils import get_logger

log = get_logger(__name__)


def method_run_name(method: MethodSpec) -> str:
    if method.type == "lora":
        return f"lora_r{method.rank}"
    return method.type


def count_trainable_parameters(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def generate_predictions(
    model, tokenizer, eval_dataset, max_new_tokens: int, max_seq_length: int, num_samples: int
) -> tuple[list[str], list[str]]:
    import torch

    was_training = model.training
    model.eval()
    predictions: list[str] = []
    references: list[str] = []
    n = min(num_samples, len(eval_dataset))
    with torch.no_grad():
        for i in range(n):
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
    single file scripts/plot_results.py can read for plotting.
    """
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    training_cfg = config.training
    run_name = f"{task}__{method_run_name(method)}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

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
    train_result = trainer.train()

    choices = get_choices(task)
    predictions, references = generate_predictions(
        model,
        tokenizer,
        eval_dataset,
        max_new_tokens=training_cfg.max_new_tokens,
        max_seq_length=training_cfg.max_seq_length,
        num_samples=training_cfg.eval_samples,
    )
    accuracy = task_accuracy(predictions, references, choices)
    trainable_params, total_params = count_trainable_parameters(model)

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
        "training_curve": training_curve_from_trainer(trainer),
    }
    write_metric(output_dir / "metrics.jsonl", metrics)
    log.info(
        "run=%s accuracy=%.4f trainable_params=%d/%d (%.4f%%)",
        run_name,
        accuracy,
        trainable_params,
        total_params,
        100 * metrics["trainable_param_pct"],
    )
    return metrics
