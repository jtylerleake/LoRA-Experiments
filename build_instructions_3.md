

Excellent. Now let's program the plots for experiment #2. We will make two plots, both are described below. Please implement. 

## Heatmap Plot: Task × Matrix Configuration
Rows = tasks (SST-2, RTE, GSM8K), columns = which matrices were adapted (W_q only, W_k only, W_v only, W_o only, W_q+W_k, W_q+W_v, all attention, MLP only, etc.). Color = performance normalized as % of full fine-tuning. This is the single most information-dense view — you can read off both which matrix matters per task and which tasks are most sensitive to the choice. Normalization is important so tasks with different absolute scales are comparable.

## Ablation Delta Plot
Start from the full-attention-LoRA condition (W_q + W_k + W_v + W_o) and plot the performance drop from removing each matrix one at a time. This is more informative than the marginal contribution plot because it controls for interactions — a matrix might look unimportant in isolation but be critical in combination. Use one panel per task so you can see whether the drop pattern is consistent or task-specific.



Terrific, now let's focus on the plots for Experiment #3: side-by-side evaluation. The plots we will want to generate are. 



Chat, you must get rid of the logs being printed in the output cells in colab, this is causing my browser to crash. Use tdqm to show progress. The output cells for each experiment run (full and mini) should just be single progress bars.