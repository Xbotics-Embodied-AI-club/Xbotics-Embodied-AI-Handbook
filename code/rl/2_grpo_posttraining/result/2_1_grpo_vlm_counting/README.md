# GRPO Small VLM Results

- Model: `unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit`
- Train data: `leonardPKU/clevr_cogen_a_train`
- Eval data: fixed `SuperCLEVR-200` counting subset

## Model Choice

The verified route uses Qwen2.5-VL 3B because the original Qwen2-VL 2B path hit current Unsloth/TRL VLM-GRPO incompatibilities: Qwen2-VL does not support Unsloth `fast_inference=True`, and the non-vLLM path fails during generation.

## Train-before vs train-after comparison

The extended run evaluates the same fixed SuperCLEVR-200 slice before and after GRPO.
Kept in this directory:

- Summary: `base_vs_adapter_summary.json`
- Adapter: `adapter/adapter_model.safetensors`

Only the two endpoint evaluations were kept; the run produced no per-step curve, so
the numbers below are before/after values, not a training trajectory.

Training:

```text
train_examples=512
max_steps=300
num_generations=4
gradient_accumulation_steps=4
train_runtime=1320.6559
train_loss=-0.0007330561888703188
train_steps_per_second=0.227
```

SuperCLEVR-200:

```text
base_accuracy=0.095
adapter_accuracy=0.44
delta_accuracy=0.345
base_format=0.22
adapter_format=0.79
delta_format=0.57
```
