"""Dataset loading/prep for experiments 1-3.

Experiment 1 (rank ablation) runs the same model/method sweep across three
task types so the rank-vs-quality story isn't specific to one kind of task:

- sst2 (GLUE): binary sentiment classification.
- rte (GLUE): binary natural-language-inference (entailment/not_entailment).
- gsm8k: grade-school math word problems, scored on the standard
  "#### <number>" final-answer format.

Every task is reduced to the same {"prompt", "completion"} shape so training
and generation-based eval (training/common.py) don't need per-task branches
beyond picking the right scorer (see eval/metrics.py's `choices` argument).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, DatasetDict

_GSM8K_PROMPT = (
    "Solve the following grade-school math problem. Show your work, then "
    "give the final answer on its own line as '#### <number>'.\n\n"
    "Question: {question}\n"
    "Answer:"
)

_SST2_PROMPT = (
    "Classify the sentiment of the following movie review as positive or "
    "negative.\n\nReview: {sentence}\nSentiment:"
)
_SST2_LABELS = {0: "negative", 1: "positive"}

_RTE_PROMPT = (
    "Does the premise entail the hypothesis? Answer with 'entailment' or "
    "'not_entailment'.\n\nPremise: {sentence1}\nHypothesis: {sentence2}\nAnswer:"
)
_RTE_LABELS = {0: "entailment", 1: "not_entailment"}


def _format_gsm8k(example: dict[str, Any]) -> dict[str, str]:
    return {
        "prompt": _GSM8K_PROMPT.format(question=example["question"]),
        "completion": " " + example["answer"].strip(),
    }


def _format_sst2(example: dict[str, Any]) -> dict[str, str]:
    return {
        "prompt": _SST2_PROMPT.format(sentence=example["sentence"].strip()),
        "completion": " " + _SST2_LABELS[example["label"]],
    }


def _format_rte(example: dict[str, Any]) -> dict[str, str]:
    return {
        "prompt": _RTE_PROMPT.format(
            sentence1=example["sentence1"].strip(), sentence2=example["sentence2"].strip()
        ),
        "completion": " " + _RTE_LABELS[example["label"]],
    }


@dataclass(frozen=True)
class _TaskSpec:
    hf_repo: str
    hf_config: str | None
    train_split: str
    eval_split: str
    formatter: Callable[[dict[str, Any]], dict[str, str]]
    # None for open-ended generation tasks (scored via eval.metrics.extract_final_answer);
    # a fixed label set for classification tasks (scored via eval.metrics.classification_accuracy).
    choices: list[str] | None


_TASKS: dict[str, _TaskSpec] = {
    "gsm8k": _TaskSpec("openai/gsm8k", "main", "train", "test", _format_gsm8k, choices=None),
    "sst2": _TaskSpec("nyu-mll/glue", "sst2", "train", "validation", _format_sst2, list(_SST2_LABELS.values())),
    "rte": _TaskSpec("nyu-mll/glue", "rte", "train", "validation", _format_rte, list(_RTE_LABELS.values())),
}


def get_choices(task: str) -> list[str] | None:
    """The fixed label set for a classification task, or None for open-ended generation."""
    return _TASKS[task].choices


def load_dataset(task: str) -> DatasetDict:
    """Load and format a registered task into {"train", "eval"} splits of {"prompt", "completion"} pairs."""
    if task not in _TASKS:
        raise NotImplementedError(f"Task {task!r} not implemented yet. Known tasks: {sorted(_TASKS)}")
    import datasets as hf_datasets

    spec = _TASKS[task]
    raw = hf_datasets.load_dataset(spec.hf_repo, spec.hf_config)
    train = raw[spec.train_split].map(spec.formatter, remove_columns=raw[spec.train_split].column_names)
    eval_ = raw[spec.eval_split].map(spec.formatter, remove_columns=raw[spec.eval_split].column_names)
    return DatasetDict({"train": train, "eval": eval_})


def tokenize_for_causal_lm(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    """Tokenize {"prompt", "completion"} pairs for causal-LM training.

    Prompt tokens are masked out (label = -100) so loss is only computed on
    the completion, matching standard instruction-tuning practice.
    """

    def _tokenize(example: dict[str, str]) -> dict[str, list[int]]:
        prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
        full_text = example["prompt"] + example["completion"] + tokenizer.eos_token
        full_ids = tokenizer(
            full_text, add_special_tokens=False, truncation=True, max_length=max_length
        )["input_ids"]
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        }

    return dataset.map(_tokenize, remove_columns=dataset.column_names)
