# CSE 151B Competition Submission

## Summary
- Model: `Qwen/Qwen3-4B-Thinking-2507`
- GPU used: NVIDIA GeForce RTX 5090
- Approximate end-to-end inference time: ~148.2 minutes
- Output file: `results/submission.csv`

## Model Weights
This submission loads the base model directly from Hugging Face.

If you are using a fine-tuned checkpoint, upload it to the Hugging Face Hub and update `MODEL_ID` in `run_inference.py` to point to your repo, for example:

```python
MODEL_ID = "your-username/your-model-name"
```

If you need to use local weights instead, place them in a directory such as `./weights/your-model-name/` and update the model path in the code accordingly.

## Environment Setup
Install the Python dependencies required by the notebook/script, then make sure your GPU runtime is available. The current pipeline was developed for vLLM + Transformers with CUDA-enabled PyTorch.

A minimal run looks like this:

```bash
python -m pip install -U vllm transformers sympy numpy tqdm bitsandbytes antlr4-python3-runtime ipykernel jupyter huggingface_hub
```

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
