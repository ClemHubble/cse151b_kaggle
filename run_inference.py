import csv
import json
import math
import os
import re
import time
from collections import Counter
from fractions import Fraction

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["HF_HUB_DISABLE_XET"] = "1"

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH = "data/private.jsonl"
OUT_PATH = "results/submission.csv"

N_SAMPLES = 1
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_TOKENS = 32768
TP_SIZE = 1

SYSTEM_PROMPT_FREE = (
    "You are an expert mathematical reasoner.\n\n"
    "ANSWER FORMAT RULES:\n"
    "1. Put ALL answers inside a single \\boxed{}, comma-separated, in the exact order "
    "the [ANS] placeholders appear. Example (3 placeholders): \\boxed{41, 35, 16}.\n"
    "2. EMBEDDED CHOICE: If an [ANS] slot is immediately followed by labeled options "
    "(e.g. '[ANS] A. Reject  B. Fail to reject', or '[ANS] A. Yes  B. No'), "
    "output ONLY the capital letter for that slot — not the option text. "
    "Mixed example: \\boxed{1.96, A, 0.032, B}.\n"
    "3. Give EXACT answers (fractions, radicals, expressions) unless rounding is explicitly "
    "requested, then use the precision specified. Reduce all fractions to lowest terms.\n"
    "4. For answers 'in terms of' a variable, use the exact variable name; do not evaluate.\n"
    "5. Do NOT include units in \\boxed{} unless the problem explicitly requires them.\n"
    "6. Output \\boxed{} immediately after your reasoning. Nothing after the box."
)

SYSTEM_PROMPT_FREE_OLYMPIAD = (
    "You are an expert mathematical reasoner solving a competition-level problem.\n\n"
    "ANSWER FORMAT RULES:\n"
    "1. Place your single final answer in \\boxed{}.\n"
    "2. Give exact answers (integers, reduced fractions, radicals) unless told to round.\n"
    "3. Commit to one approach and execute it fully. If you find a flaw, correct inline "
    "and continue — do not restart from scratch.\n"
    "4. Output \\boxed{} immediately after your reasoning. Nothing after the box."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematical reasoner.\n\n"
    "ANSWER FORMAT RULES:\n"
    "1. Output ONLY the capital letter of the correct option inside \\boxed{}. "
    "Options run A through J. Example: \\boxed{G}.\n"
    "2. Write the letter only — do NOT reproduce the option text.\n"
    "3. Treat ALL options as valid candidates, including 'Unable to determine', "
    "'None of the above', and 'Unchanged'.\n"
    "4. Output \\boxed{} immediately after your reasoning. Nothing after the box."
)


def extract_boxed(text: str) -> str | None:
    """Brace-balanced extraction of the last \\boxed{...}."""
    results = []
    pos = 0
    while True:
        start = text.find(r"\boxed{", pos)
        if start == -1:
            break
        depth = 0
        content_start = start + len(r"\boxed{")
        end = -1
        for j in range(content_start - 1, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end != -1:
            results.append(text[content_start:end].strip())
            pos = end + 1
        else:
            results.append(text[content_start:].strip())
            break
    return results[-1] if results else None


def latex_to_numeric(s: str) -> str | None:
    """Resolves common LaTeX numeric expressions to plain numbers."""
    s = s.strip()
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        try:
            fraction = Fraction(int(m.group(1)), int(m.group(2)))
            value = float(fraction)
            return str(int(value)) if value == int(value) else str(fraction)
        except Exception:
            pass
    m = re.fullmatch(r"\\sqrt\{(\d+)\}", s)
    if m:
        value = math.sqrt(int(m.group(1)))
        return str(int(value)) if value == int(value) else f"{value:.6g}"
    m = re.fullmatch(r"(-?[\d.]+)\s*\\times\s*10\^\{(-?\d+)\}", s)
    if m:
        try:
            value = float(m.group(1)) * 10 ** int(m.group(2))
            return str(int(value)) if value == int(value) else f"{value:.6g}"
        except Exception:
            pass
    return None


def normalize(ans: str) -> str:
    """Canonical form for stable comparison."""
    if not isinstance(ans, str):
        return ""

    s = ans.strip()
    s = s.replace("\\$", "").strip("$").strip()
    s = s.replace("\\,", "").replace("\\ ", "").replace("\\!", "").strip()

    resolved = latex_to_numeric(s)
    if resolved is not None:
        return resolved

    m = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", s)
    if m:
        try:
            fraction = Fraction(int(m.group(1)), int(m.group(2)))
            value = float(fraction)
            if math.isinf(value) or math.isnan(value):
                return str(fraction)
            return str(int(value)) if value == int(value) else str(fraction)
        except Exception:
            pass

    try:
        value = float(s.replace(",", ""))
        if math.isinf(value) or math.isnan(value):
            return s.lower()
        return str(int(value)) if value == int(value) else f"{value:.6g}"
    except (ValueError, OverflowError):
        pass

    if re.fullmatch(r"[a-jA-J]", s):
        return s.upper()

    if "," in s:
        return ", ".join(normalize(part.strip()) for part in s.split(","))

    return s.lower()


def fallback_mcq(text: str) -> str:
    tail = text[-400:]
    match = re.search(r"\b(?:answer\s+is\s+|option\s+|choice\s+)?([A-J])[\.\)]?\s*$", tail, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    letters = re.findall(r"\b([A-J])\b", tail)
    return letters[-1].upper() if letters else "A"


def fallback_numeric(text: str, ans_count: int) -> str:
    tail = text[-500:]
    nums = re.findall(r"-?\d+(?:\.\d+)?", tail)
    if nums:
        if ans_count > 1 and len(nums) >= ans_count:
            return ", ".join(nums[-ans_count:])
        return nums[-1]
    return ", ".join(["0"] * max(1, ans_count))


def extract_answer(text: str, is_mcq: bool, ans_count: int) -> str:
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    return fallback_mcq(text) if is_mcq else fallback_numeric(text, ans_count)


def majority_vote(answers: list[str]) -> str:
    normed = [normalize(answer) for answer in answers if answer]
    if not normed:
        return answers[0] if answers else "0"
    winner = Counter(normed).most_common(1)[0][0]
    for answer in answers:
        if normalize(answer) == winner:
            return answer
    return answers[0]


def get_system_prompt(question: str, options) -> str:
    if options is not None:
        return SYSTEM_PROMPT_MCQ
    if question.count("[ANS]") == 0:
        return SYSTEM_PROMPT_FREE_OLYMPIAD
    return SYSTEM_PROMPT_FREE


def build_prompts(data, tokenizer):
    prompts = []
    for item in data:
        options = item.get("options")
        question = item["question"]
        system = get_system_prompt(question, options)

        if options:
            labels = [chr(65 + i) for i in range(len(options))]
            opts_text = "\n".join(f"{label}. {option.strip()}" for label, option in zip(labels, options))
            user = f"Problem:\n{question}\n\nAnswer Choices:\n{opts_text}\n\nSelect the correct letter."
        else:
            user = f"Problem:\n{question}\n\nSolve completely and provide your final answer in \\boxed{{}}."

        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )

        prompts.append(prompt)

    return prompts


def run_inference(
    model_id: str = MODEL_ID,
    data_path: str = DATA_PATH,
    output_path: str = OUT_PATH,
    n_samples: int = N_SAMPLES,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = MAX_TOKENS,
    tp_size: int = TP_SIZE,
):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(data_path, "r", encoding="utf-8") as handle:
        data = [json.loads(line) for line in handle]

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    prompts = build_prompts(data, tokenizer)

    print(f"Loading model onto GPU...")
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.90,
        max_model_len=max_tokens,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
    )

    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    t0 = time.time()
    print(f"Generating {n_samples} traces x {len(prompts)} problems...")
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    elapsed_minutes = (time.time() - t0) / 60

    missing_boxes = 0
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "response"], quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for index, out in enumerate(outputs):
            item = data[index]
            is_mcq = bool(item.get("options"))
            ans_count = 1 if is_mcq else item["question"].count("[ANS]")

            texts = [candidate.text.strip() for candidate in out.outputs]
            answers = [extract_answer(text, is_mcq, ans_count) for text in texts]

            for text in texts:
                if extract_boxed(text) is None:
                    missing_boxes += 1

            final_answer = majority_vote(answers)

            representative = texts[0]
            for text, answer in zip(texts, answers):
                if normalize(answer) == normalize(final_answer):
                    representative = text
                    break

            response = representative + f"\n\n\\boxed{{{final_answer}}}"
            writer.writerow({"id": item["id"], "response": response})

    gpu_name = os.environ.get("CUDA_VISIBLE_DEVICES", "GPU")
    print(f"Saved -> {output_path}")
    print(f"Fallbacks (no \\boxed): {missing_boxes} / {len(outputs) * n_samples} traces")
    print(f"Elapsed: {elapsed_minutes:.1f} minutes")
    print(f"GPU: {gpu_name}")
    return output_path


if __name__ == "__main__":
    run_inference()
