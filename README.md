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
python scripts/run_experiment.py --config src/lora_experiments/config/experiment_1_rank_ablation.yaml --device cpu --dry-run
```

## Adding a dependency

1. Add it to `docker/requirements/base.in`.
2. Run `pwsh scripts/compile_requirements.ps1` to regenerate the locked
   `base.txt` (used by Docker) and `colab.txt` (used by Colab notebooks).
3. Commit both `base.in` and the regenerated `.txt` files.

## Adding a model

Add an entry to `src/lora_experiments/config/models.yaml` with its
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
src/lora_experiments/
  config/        experiment YAML configs + the model registry
  data/          dataset loaders (incl. ARC-AGI-style loader for exp 4)
  models/        model/tokenizer registry
  training/      lora.py, adapters.py, prefix_tuning.py, full_finetune.py, test_time_tuning.py
  eval/          metrics
  utils/         logging, seeding, Colab/Drive helpers
notebooks/       Colab notebooks (bootstrap + one per experiment)
scripts/         run_experiment.py, compile_requirements.ps1, sync_outputs.py
outputs/         gitignored local mirror of metrics/logs pulled from Drive
tests/           unit tests run inside the Docker dev image
docs/            colab_workflow.md
```

## Status

Infrastructure scaffold only — training logic in `src/lora_experiments/training/`
and `scripts/sync_outputs.py` are stubs (`NotImplementedError`) pending the
first real experiment implementation. `--dry-run` validates the full
config → model-registry → method-dispatch → output-dir wiring today.

## Note on the LoRA library

The original `microsoft/LoRA` repo is used as the algorithm reference (see
`docs/colab_workflow.md`), but the runtime dependency is Hugging Face
**PEFT** — it's actively maintained, integrates with `transformers`, and its
`target_modules` config is what Experiment 2 sweeps over directly.
