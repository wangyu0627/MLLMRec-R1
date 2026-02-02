import os
import json
import time
import random
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

from tqdm import tqdm
from openai import OpenAI

# ============================================================
# Configuration
# ============================================================
ROOT = "/root/autodl-tmp/RL4MSRec"
DATASET = "microlens"

PSEUDO_COT_JSON = f"{ROOT}/data/{DATASET}/{DATASET}_pseudo-cot.json"
CAPTION_JSON = f"{ROOT}/data/{DATASET}/{DATASET}_captions.json"
SAVE_JSON = f"{ROOT}/data/{DATASET}/{DATASET}_deepseek_cot.json"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-xxxxxxxx")
DEEPSEEK_BASE_URL = "Your API interface"
DEEPSEEK_MODEL = "deepseek-r1-0528"

# Concurrency controls
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "16"))      # thread pool size
MAX_INFLIGHT = int(os.getenv("MAX_INFLIGHT", "8"))     # max concurrent API calls
REQUEST_INTERVAL = float(os.getenv("REQUEST_INTERVAL", "0.0"))  # optional per-task sleep

# Retry controls
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
BACKOFF_BASE = float(os.getenv("BACKOFF_BASE", "0.8"))  # seconds


# ============================================================
# JSON Utilities
# ============================================================
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ============================================================
# Prompt Builders (same logic as your current script)
# ============================================================
def get_caption_block(item_id: str, captions: Dict[str, Dict[str, Any]]) -> str:
    """
    Return formatted caption + step template for a given item.

    captions json must follow the structure:
    {
        "item_id": {
            "title": "...",
            "caption": "...",
            "steps": ["Step1 ...", "Step2 ...", ...]
        }
    }
    """
    item_id = str(item_id)
    if item_id not in captions:
        return "Caption: None\nSteps: None"

    c = captions[item_id]
    lines = [f"Caption: {c.get('caption', '')}"]

    steps = c.get("steps", None)
    if steps:
        lines.append("Reasoning Steps Template From Caption:")
        for i, s in enumerate(steps, 1):
            lines.append(f"- Step {i}: {s}")

    return "\n".join(lines)

def build_prompt(
    uid: str,
    hist: List[Dict[str, Any]],
    tgt: Dict[str, Any],
    captions: Dict[str, Dict[str, Any]],
    pseudo_cot_reasoning: Optional[str] = None,
) -> str:
    lines: List[str] = []

    lines.append(
        f"User ID: {uid}\n"
        f"We provide titles, captions, and reasoning templates extracted from images.\n"
        f"Now generate final reasoning WITHOUT image input, using captions instead."
    )

    for idx, item in enumerate(hist, 1):
        item_id = str(item["item_id"])
        title = item.get("title", "")
        lines.append(f"\nHistory Item {idx}:")
        lines.append(f"Title: {title} (id={item_id})")
        lines.append(get_caption_block(item_id, captions))

    # Note: we keep your existing fields, but the instruction below does NOT assume target is known.
    tid = str(tgt["item_id"])
    title = tgt.get("title", "")
    lines.append(f"\nFinal Target Item:")
    lines.append(f"Title: {title} (id={tid})")
    lines.append(get_caption_block(tid, captions))

    if pseudo_cot_reasoning:
        lines.append(
            "We also provide a pseudo chain-of-thought reasoning previously generated:\n"
            "(This reasoning may be noisy. You should learn its structure and improve clarity while keeping logic consistent.)"
        )
        lines.append(pseudo_cot_reasoning.strip())

    # Predictive, candidate-free instruction (your latest version)
    lines.append(
        "Reason about the user's NEXT interaction based only on the history captions.\n\n"
        "Step1: Extract the most discriminative preference signals from the history captions.\n"
        "Step2: Infer the most likely next-step intent and describe what the next item should look like.\n"
        "Step3: Provide 3–5 retrieval criteria (keywords/visual cues/themes/mood/genre) that would best retrieve the true next item.\n"
        "Step4: Give a short summary of the predicted next item profile in one sentence.\n\n"
        "Rules:\n"
        "- Focus on NEXT interaction intent, not past summary.\n"
        "- Output MUST be written as consecutive plain-text lines starting with Step 1:, Step 2:, Step 3:, and Step 4:, exactly in this format.\n"
        "- Reason in a predictive, forward-looking manner.\n"
        "- Write 4–6 fluent English sentences.\n"
    )

    return "\n".join(lines)


# ============================================================
# DeepSeek Call (thread-safe usage)
# ============================================================
def deepseek_call(prompt: str) -> str:
    """
    Create a client inside the thread to avoid shared-state issues.
    """
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


def call_with_retry(prompt: str) -> str:
    """
    Simple retry with exponential backoff + jitter.
    """
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return deepseek_call(prompt)
        except Exception as e:
            last_err = e
            if attempt >= MAX_RETRIES:
                break
            sleep_s = (BACKOFF_BASE * (2 ** attempt)) + random.uniform(0.0, 0.3)
            time.sleep(sleep_s)
    raise last_err


# ============================================================
# Worker
# ============================================================
def process_one(
    idx: int,
    sample: Dict[str, Any],
    captions: Dict[str, Dict[str, Any]],
    sem: Semaphore,
) -> Tuple[int, Dict[str, Any]]:
    """
    Returns (idx, result_dict) so we can place it back in order.
    """
    with sem:  # cap inflight requests
        uid = sample["user_id"]
        hist = sample["history_items"]
        tgt = sample["target_item"]

        pseudo_cot_reasoning = sample.get("reasoning", "")

        prompt = build_prompt(uid, hist, tgt, captions, pseudo_cot_reasoning=pseudo_cot_reasoning)
        reasoning = call_with_retry(prompt)

        if REQUEST_INTERVAL > 0:
            time.sleep(REQUEST_INTERVAL)

        result = {
            "line_idx": sample.get("line_idx"),
            "user_id": uid,
            "history_items": hist,
            "target_item": tgt,
            "reasoning": reasoning,
        }
        return idx, result


# ============================================================
# Main
# ============================================================
def main(max_samples: Optional[int] = None):
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY is empty. Please export it before running.")

    pseudo = load_json(PSEUDO_COT_JSON)
    captions = load_json(CAPTION_JSON)

    data = pseudo[:max_samples] if max_samples else pseudo
    results: List[Optional[Dict[str, Any]]] = [None] * len(data)

    sem = Semaphore(MAX_INFLIGHT)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(process_one, i, sample, captions, sem): i
            for i, sample in enumerate(data)
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating (multi-thread)"):
            i = futures[fut]
            try:
                idx, res = fut.result()
                results[idx] = res
            except Exception as e:
                # Keep an error record instead of crashing the whole run
                results[i] = {
                    "line_idx": data[i].get("line_idx"),
                    "user_id": data[i].get("user_id"),
                    "history_items": data[i].get("history_items"),
                    "target_item": data[i].get("target_item"),
                    "reasoning": "",
                    "error": repr(e),
                }

    # Filter out Nones (should not happen) and save
    final_results = [r for r in results if r is not None]
    save_json(final_results, SAVE_JSON)
    print("FINISHED & SAVED ->", SAVE_JSON)


if __name__ == "__main__":
    # For debugging, set a small number; set None for full dataset.
    main(max_samples=10000)
