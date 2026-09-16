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
3. **Get the code onto the runtime** from the public GitHub remote:
   ```bash
   !git clone https://github.com/jtylerleake/LoRA-Experiments.git /content/lora_experiments
   %cd /content/lora_experiments
   ```
4. **Install dependencies on top of Colab's preinstalled, CUDA-matched
   torch** — do not `pip install torch` here, it can break Colab's CUDA
   setup. Also remove Colab's preinstalled `torchao` (a quantization
   library we don't use — our quantization backend is bitsandbytes):
   `peft`'s LoRA module dispatcher probes every optional backend including
   torchao, and its version check *raises* instead of skipping when
   torchao is present but older than that `peft` release expects,
   crashing `get_peft_model()` entirely on Colab's default image.
   ```bash
   !pip install -q -r docker/requirements/colab.txt
   !pip install -q -e .
   !pip uninstall -y -q torchao
   ```
5. **Run an experiment**, pointing output at Drive:
   ```bash
   !python scripts/run_experiment.py \
       --config src/config/experiment_1_rank_ablation.yaml \
       --device cuda \
       --output-root /content/drive/MyDrive/lora_experiments_outputs
   ```

`notebooks/colab_bootstrap.ipynb` contains exactly these cells; each
`notebooks/expN_*.ipynb` repeats the pattern pointed at its own config.

## Smoke-testing before a full sweep

Before committing GPU time to the full `experiment_1_rank_ablation.yaml`
sweep (3 tasks x 7 methods, can run for hours), run
`notebooks/exp1_rank_ablation_mini.ipynb` first —
`experiment_1_rank_ablation_mini.yaml` has the exact same task/method
structure but caps training to ~32 examples and 1 epoch per run, so all 21
runs finish in a couple of minutes. It writes to a separate
`exp1_rank_ablation_mini` output folder, so it never touches the full run's
results, and its own `metrics.jsonl` + `plot_results.py` output let you
confirm training, metrics, and plotting are all wired correctly before
starting the real sweep. Accuracy numbers from the mini run are meaningless
(barely any training data) — only the wiring matters.

## Plotting results

`scripts/plot_results.py --metrics <path/to/metrics.jsonl>` reads a run's
metrics and writes three plots built with seaborn: accuracy vs. trainable
params, accuracy vs. rank, and a single combined training-curves plot
overlaying every task/rank/method run (x-axis normalized to % of training
complete, y-axis log-scaled, since different tasks can have very different
dataset sizes and loss magnitudes). Since Drive is already mounted in
Colab, it's simplest to run this directly against the Drive-mounted
`metrics.jsonl` there (see the cell in `notebooks/exp1_rank_ablation.ipynb`)
rather than syncing first. It reads whatever rows are in `metrics.jsonl`
regardless of how many separate `run_experiment.py` invocations wrote
them — see "Running experiment 1-A piecemeal" below.

## Running experiment 1-A piecemeal

`experiment_1a_rank_ablation.yaml` is experiment 1 with the full
fine-tuning baseline removed (LoRA ranks only). Since the full sweep is
still large, `notebooks/exp1a_rank_ablation.ipynb` splits it into three
cells — one per task — using `run_experiment.py --task <name>` to restrict
a single invocation to just that task:
```bash
!python scripts/run_experiment.py \
    --config src/config/experiment_1a_rank_ablation.yaml \
    --task sst2 \
    --device cuda \
    --output-root /content/drive/MyDrive/lora_experiments_outputs
```
Every invocation appends to the same `metrics.jsonl`, so the three cells
can run in the same session or three separate ones (if Colab disconnects
partway through, you only lose the currently-running task, not the whole
sweep) — the notebook's final plotting cell reads the shared file and
aggregates across whichever tasks have completed so far, the same as if
they'd all run in one invocation. `experiment_1a_rank_ablation_mini.yaml`
+ `notebooks/exp1a_rank_ablation_mini.ipynb` smoke-test this same
per-task-split-then-aggregate flow in a couple of minutes before
committing to the full sweep.

## Getting results back locally

Checkpoints stay in Drive (they're large). `metrics.jsonl` is a small
JSON-lines file written alongside them — `scripts/sync_outputs.py` (still a
stub; the intended approach is `rclone` against Drive) is meant to pull just
that back into the local, gitignored `outputs/` tree. Until it's
implemented, downloading `metrics.jsonl` from Drive by hand is a fine
substitute — it's the only artifact worth syncing.

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
  (`src/training/`) instead of four bespoke ones.

## Model sizing on Colab Pro

Colab Pro GPUs are typically T4 (16GB), L4 (24GB), or A100 (40GB), assigned
dynamically — you don't pick the exact GPU. The model registry
(`src/config/models.yaml`) reflects that:

- Qwen2.5 0.5B/1.5B/7B-Instruct run unquantized on any of those GPUs.
- `gpt-oss-20b` is loaded 4-bit (bitsandbytes) to fit comfortably even on a
  T4/L4.

## Iterating on Colab

Since Colab clones from GitHub, a local code change isn't visible on Colab
until it's pushed to `origin/master` — `!git pull` (or re-clone) at the top
of a Colab session picks up new commits.
