"""Bottleneck adapter training, for the experiment 3 comparison.

Hugging Face PEFT (see docs/colab_workflow.md) doesn't include classic
Houlsby-style bottleneck adapters as a first-class method — checked
directly against the installed `peft` (0.21.0 at the time of writing): it
has LoRA, prefix/prompt tuning, IA3, and a long tail of newer
reparameterization methods, but no plain down-project/up-project adapter.
So this module implements one directly: a small residual adapter
(down-project -> GELU -> up-project, zero-initialized so training starts
as a no-op) inserted via a forward hook after every decoder layer. Only
the adapter parameters are trainable; the base model is frozen.

Only tested against the Qwen2 architecture family (the only models in
config/models.yaml) — relies on `base_model.model.layers` being the
decoder stack, a LLaMA/Qwen2-style convention, not a universal one.
"""
from __future__ import annotations

from torch import Tensor, nn

from config.schema import MethodSpec

_DEFAULT_BOTTLENECK_SIZE = 64


class BottleneckAdapter(nn.Module):
    """down-project -> GELU -> up-project, added residually.

    `up_proj` is zero-initialized so the adapter is an identity function at
    the start of training (standard adapter-tuning practice) — training
    then learns how much to deviate from the frozen base model.
    """

    def __init__(self, hidden_size: int, bottleneck_size: int):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return hidden_states + self.up_proj(self.act(self.down_proj(hidden_states)))


def build_adapter_model(base_model, method: MethodSpec):
    """Freeze `base_model` and insert one BottleneckAdapter per decoder
    layer via a forward hook.
    """
    bottleneck_size = method.bottleneck_size or _DEFAULT_BOTTLENECK_SIZE

    for param in base_model.parameters():
        param.requires_grad_(False)

    ref_param = next(base_model.parameters())
    hidden_size = base_model.config.hidden_size

    adapters = nn.ModuleList(
        BottleneckAdapter(hidden_size, bottleneck_size).to(device=ref_param.device, dtype=ref_param.dtype)
        for _ in base_model.model.layers
    )

    def make_hook(adapter: BottleneckAdapter):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                return (adapter(output[0]), *output[1:])
            return adapter(output)

        return hook

    for layer, adapter in zip(base_model.model.layers, adapters):
        layer.register_forward_hook(make_hook(adapter))

    # Setting an nn.ModuleList as an attribute registers it as a submodule
    # (PyTorch's nn.Module.__setattr__ handles this), so its params show up
    # in base_model.parameters() — and thus the Trainer's optimizer — without
    # a custom training loop.
    base_model.bottleneck_adapters = adapters
    return base_model
