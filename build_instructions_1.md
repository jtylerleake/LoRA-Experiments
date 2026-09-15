
# Purpose
Claude, this is an empty folder called lora_experiments. I want to write code, build dockerfiles, create images in docker, generate outputs, etc. I need help building this infrastructure. The purpose of this repository is to experiment with LoRA: a method for fine-tuning LLMs using low rank matrices. 

## Requirements
- Some of the software we will need for this project include basic libraries like numpy, pytorch, seaborn, etc.; open source models (e.g., open-ai, qwen, kiwi); and of course the open source lora library from microsoft: https://github.com/microsoft/lora. 
- Of course, I don't want to install any of this software locally on my machine. I want to leverage Docker completely. Do not download large packages and models locally to my computer. 

## Experiments 
You don't need to implement the experiments yet, but you should have them in mind for design purposes.

- Experiment 1: Rank Ablation: we're going to fine-tune the same model on the same task with different variants of rank (1,2,4,8,16,64) and full fine-tuning. 
- Experiment 2: Matrix application study: since LoRA can be applied to any subset of weights in the model/transformer, we're going to experiment with applying LoRA to different parts independently (e.g., just the query matrix, or just query+value matrix)
- Experiment 3: Side-by-side comparison: using the same model, we will compare full re-training, fine-tuning with adaptation layers, prefix tuning, and LoRA.
- Experiment 4: Test-time-tuning with LoRA for ARC-AGI style tasks: use LoRA to update weights at inference time on few-shot learning tasks and measure performance.


## Google Colab for Training
- I have a Google Colab pro subscription.
- Even though I'm building my code locally and running with Docker, I want the actual compute to be handled with Google Colab CPUs/GPUs when possible to offload the compute from my machine. 
- I want to connect remotely to them and run the code, which I have a Pro subscription for.

## First Task
- Your first task is to build a plan for this project, including how we'll manage compute (where, whatm, how); build and run instructions; dockerfile creation; structure of our experiment code; where we'll store outputs; directory structure, etc.
