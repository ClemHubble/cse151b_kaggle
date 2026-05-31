# CSE 151B Competition Submission

## Summary
- Model: `Qwen/Qwen3-4B-Thinking-2507`
- GPU used: NVIDIA GeForce RTX 5090
- Approximate end-to-end inference time: ~148.2 minutes
- Output file: `results/submission.csv`

## Model Weights
This submission loads the base model directly from Hugging Face.

## Reproduce Results
The single entry point is `run_inference()` in `run_inference.py`.

```python
from run_inference import run_inference

run_inference(
    model_id="Qwen/Qwen3-4B-Thinking-2507",
    data_path="data/private.jsonl",
    output_path="results/submission.csv",
)
```

Or from the command line:

```bash
python run_inference.py
```

## Hyperparameters
The final submission settings are defined in `run_inference.py` and mirrored in the notebook:
- `N_SAMPLES = 1`
- `TEMPERATURE = 0.6`
- `TOP_P = 0.95`
- `MAX_TOKENS = 32768`
- `TP_SIZE = 1`

These are the values used for the reported submission run.
