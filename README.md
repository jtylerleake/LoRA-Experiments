# lora_experiments

Infrastructure for experimenting with LoRA (Low-Rank Adaptation) fine-tuning:
rank ablation, matrix-application study, method comparison (full FT vs.
adapters vs. prefix tuning vs. LoRA), and test-time LoRA tuning on
ARC-AGI-style few-shot tasks.

## Architecture at a glance

- **Docker is the dependency source of truth and the local execution
  environment** — used for editing, linting, unit tests, and CPU "dry runs"
  that validate an experiment config end-to-end before spending real GPU
  time. No model weights or datasets are ever downloaded to your machine.
- **Google Colab Pro is the GPU compute environment.** Standard Colab
  runtimes can't run arbitrary Docker containers with GPU passthrough, so
  Colab notebooks install a generated, Colab-compatible subset of the same
  pinned requirements (`docker/requirements/colab.txt`) instead of running
  the Docker image directly.
- **One CLI, two environments.** `scripts/run_experiment.py` is the single
  entrypoint used both locally (`--device cpu --dry-run`) and in Colab
  (`--device cuda`) — the code path never forks between "local" and "real."
- **Outputs live in Google Drive** during a run (checkpoints, logs,
  metrics); `scripts/sync_outputs.py` pulls just the small metrics/logs back
  locally for plotting with seaborn/matplotlib, without ever downloading
  checkpoints.

See `docs/colab_workflow.md` for the step-by-step Colab flow.

## Local development (Docker)

```powershell
# Build + open a shell in the dev container
docker compose -f docker/docker-compose.yml build dev
docker compose -f docker/docker-compose.yml run --rm dev bash

# Inside the container:
ruff check .
pytest
python scripts/run_experiment.py --config src/config/experiment_1_rank_ablation.yaml --device cpu --dry-run
```

## Adding a dependency

1. Add it to `docker/requirements/base.in`.
2. Run `pwsh scripts/compile_requirements.ps1` to regenerate the locked
   `base.txt` (used by Docker) and `colab.txt` (used by Colab notebooks).
3. Commit both `base.in` and the regenerated `.txt` files.

## Adding a model

Add an entry to `src/config/models.yaml` with its
Hugging Face repo id and quantization hint — no code changes required
unless the model needs bespoke loading logic.

## Running on Colab

See `docs/colab_workflow.md`. Short version: open
`notebooks/colab_bootstrap.ipynb` (or one of the `expN_*.ipynb` notebooks,
which include the same setup cells), select a GPU runtime, run top to
bottom.

## Project layout

```
docker/          Dockerfile, docker-compose, pinned requirements
src/
  config/        experiment YAML configs + the model registry
  data/          dataset loaders (incl. ARC-AGI-style loader for exp 4)
  models/        model/tokenizer registry
  training/      lora.py, adapters.py, prefix_tuning.py, full_finetune.py, test_time_tuning.py
  eval/          metrics
  utils/         logging, seeding, Colab/Drive helpers
notebooks/       Colab notebooks (bootstrap + one per experiment)
scripts/         run_experiment.py, plot_results.py, plot_matrix_study_results.py,
                 compile_requirements.ps1, sync_outputs.py
outputs/         gitignored local mirror of metrics/logs pulled from Drive
tests/           unit tests run inside the Docker dev image
docs/            colab_workflow.md
```

## Status

Experiments 1-3 are implemented end-to-end on one model
(`qwen2.5-0.5b-instruct`): dataset loading/formatting for SST-2, RTE, and
GSM8K; model loading with optional quantization; LoRA (`peft.LoraConfig`),
full-fine-tune, PEFT prefix-tuning, and a from-scratch bottleneck-adapter
(PEFT has no classic adapter method) builder; a shared train/eval loop
(`training/common.py`) that trains, scores accuracy per task, and writes
`metrics.jsonl`. `--dry-run` validates config → model-registry →
method-dispatch → output-dir wiring without touching real weights.
Two seaborn plotting scripts read a run's `metrics.jsonl`:
`scripts/plot_results.py` (experiments 1/1-A: accuracy vs. trainable
params, accuracy vs. rank, one training-curves plot per task) and
`scripts/plot_matrix_study_results.py` (experiment 2: a task x
matrix-configuration heatmap of raw eval accuracy, and a per-task
ablation-delta plot for removing each attention matrix from the
full-attention-LoRA condition).

Full fine-tuning is dropped everywhere except experiment 3 (where it's a
direct comparison point) — it's by far the most expensive method to run,
and every experiment below is also otherwise scaled down for compute cost
(1 epoch, fewer eval samples, a capped SST-2 split, shorter generations
for classification tasks):

- **Experiment 1** (rank ablation): 3 tasks x 3 LoRA ranks (1, 8, 64) = 9 runs.
- **Experiment 1-A**: same as experiment 1, run one task at a time via
  `run_experiment.py --task <name>` so a sweep can be split across several
  Colab sessions and still aggregate into one `metrics.jsonl` — see
  `docs/colab_workflow.md`.
- **Experiment 2** (matrix application study): 3 tasks x 12 LoRA
  target-module variants (individual matrices, pairs, all-attention,
  MLP-only, and 4 leave-one-out-from-full-attention configs) = 36 runs.
- **Experiment 3** (method comparison): full-FT, bottleneck adapters, prefix tuning, LoRA on SST-2.
- Every experiment above has a `_mini` config + notebook (a handful of
  training examples, 1 epoch) that smoke-tests the full pipeline in
  minutes before committing GPU time to the real sweep.

**Experiment 4** (test-time tuning on ARC-AGI-style tasks) is still a stub:
`data/arc_agi.py` and `training/test_time_tuning.py` raise
`NotImplementedError`. `scripts/sync_outputs.py` (pulling metrics back from
Drive automatically) is also still a stub — downloading `metrics.jsonl` by
hand from Drive is the current substitute.

## Note on the LoRA library

The original `microsoft/LoRA` repo is used as the algorithm reference (see
`docs/colab_workflow.md`), but the runtime dependency is Hugging Face
**PEFT** — it's actively maintained, integrates with `transformers`, and its
`target_modules` config is what Experiment 2 sweeps over directly.
