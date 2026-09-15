# Colab workflow

## Why Docker doesn't run inside Colab

Standard Google Colab runtimes (including Colab Pro) don't provide
privileged/nested-virtualization access, so you can't `docker build`/`docker
run` a container with GPU passthrough inside a Colab notebook. Docker is
therefore used only for the local, CPU-side dev environment (editing,
linting, tests, dry runs). GPU training happens in a plain Colab Python
environment that installs the same pinned dependencies from
`docker/requirements/colab.txt`.

## One-time-per-session steps (every notebook does this)

1. **Select a GPU runtime**: `Runtime > Change runtime type > T4/L4/A100 GPU`.
2. **Mount Drive** — this is where checkpoints/logs/metrics persist across
   Colab's ephemeral runtimes:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. **Get the code onto the runtime.** Until a git remote exists for this
   repo, upload/unzip it to `/content/lora_experiments`. Once a remote is
   set up (a separate decision — see below), replace this with:
   ```bash
   !git clone <REMOTE_URL> /content/lora_experiments
   %cd /content/lora_experiments
   ```
4. **Install dependencies on top of Colab's preinstalled, CUDA-matched
   torch** — do not `pip install torch` here, it can break Colab's CUDA
   setup:
   ```bash
   !pip install -q -r docker/requirements/colab.txt
   !pip install -q -e .
   ```
5. **Run an experiment**, pointing output at Drive:
   ```bash
   !python scripts/run_experiment.py \
       --config src/lora_experiments/config/experiment_1_rank_ablation.yaml \
       --device cuda \
       --output-root /content/drive/MyDrive/lora_experiments_outputs
   ```

`notebooks/colab_bootstrap.ipynb` contains exactly these cells; each
`notebooks/expN_*.ipynb` repeats the pattern pointed at its own config.

## Getting results back locally

Checkpoints stay in Drive (they're large). Metrics/logs are small
JSON-lines/CSV files written alongside them —
`python scripts/sync_outputs.py --experiment <name> --run-id <id>` pulls
just those into the local, gitignored `outputs/` tree so they can be
plotted with seaborn/matplotlib inside the Docker dev container.

## Why Hugging Face PEFT instead of `microsoft/LoRA`

The original brief pointed at
[`microsoft/LoRA`](https://github.com/microsoft/LoRA), the reference
implementation from the LoRA paper. It's used here as the algorithm
reference/citation, but the runtime dependency is Hugging Face **PEFT**:

- It's actively maintained and integrates directly with `transformers` /
  `accelerate`, which everything else in this repo already depends on.
- Its `LoraConfig(target_modules=...)` is exactly the mechanism Experiment 2
  (matrix-application study) sweeps over — applying LoRA to just `q_proj`,
  or `q_proj`+`v_proj`, etc. — without writing custom module-patching code.
- It also implements prefix tuning and other PEFT methods out of the box,
  so Experiment 3's side-by-side comparison (full FT / adapters / prefix
  tuning / LoRA) shares one consistent code path
  (`src/lora_experiments/training/`) instead of four bespoke ones.

## Model sizing on Colab Pro

Colab Pro GPUs are typically T4 (16GB), L4 (24GB), or A100 (40GB), assigned
dynamically — you don't pick the exact GPU. The model registry
(`src/lora_experiments/config/models.yaml`) reflects that:

- Qwen2.5 0.5B/1.5B/7B-Instruct run unquantized on any of those GPUs.
- `gpt-oss-20b` is loaded 4-bit (bitsandbytes) to fit comfortably even on a
  T4/L4.

## Open item: git remote

This repo isn't pushed to a remote yet. A remote (e.g. GitHub) would let
Colab `git clone`/`git pull` instead of re-uploading a zip each session —
worth setting up once you're ready, since creating/pushing to a remote is a
separate, explicit step from this local scaffold.
