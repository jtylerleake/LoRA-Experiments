
## Git Repository Creation
- Chat, I went ahead and made this repository a Git repo and published to my GitHub account publically. 
- Please account for this change in the code and control flow with Docker/Google Colab

## Experiment 1
- Please implement experiment 1. 

#### Experiment 1 Models
- For computational purposes, we'll want to use small open source models for the experiment.
- Implement the experiment with 1 model to start. I 'll leave it up to you to pick which one, but try to use sources like openai, qwen, gemma, tinyllama, etc. 
- Later on, we'll experiment with more models simultaneously.

#### Experiment 1 Tasks
- For each rank setting and for each model used, we will want to evaluate on the following set of tasks.
1. SST-2: binary classification for movie reviews.
2. RTE: natural  language inference.
3. GSM8K: math word problems.

#### Experiment 1 Design
- The purpose of the experiment is to apply LoRA to these models with increasing matrix ranks. Use the following settings for r: 1,2,4,8,16,64. 
- The experiments shall be run for each model, for each rank setting, and for each task. E.g., for 1 models, 3 tasks, and 6 ranks, we'll generate 18 runs. 

#### Experiment 1 Outputs
- Generate neat, professional seaborn plots for each run. 
- Plot #1: performance (Y-axis) vs. log scale trainable parameters (X-axis). One curve per task. 
- Plot #2: perfornace (Y-axis) vs. rank 
- Plot #3: training curves (one per task). For each task, overlay training curves for each model/rank setting. 
