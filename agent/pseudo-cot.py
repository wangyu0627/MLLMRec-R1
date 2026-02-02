import os
import csv
import json
from typing import List, Optional

import torch
from tqdm import tqdm
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor

# ================== Configuration ==================
# Modify according to your project path
ROOT = "/root/autodl-tmp/MLLMRec-R1"
DATASET = "microlens"

MODEL_ID = f"{ROOT}/Qwen3-VL-8B-Instruct"
IMG_DIR = f"{ROOT}/data/{DATASET}/images"
TITLE_FILE = f"{ROOT}/data/{DATASET}/{DATASET}_titles.csv"
TRAIN_TSV = f"{ROOT}/data/{DATASET}/train.tsv"
SAVE_JSON = f"{ROOT}/data/{DATASET}_pseudo-cot.json"

K = 10       # Most recent K interactions (the last one is the target)

# ================== Utility Functions ==================
def load_titles(title_file: str) -> dict:
    """Load item -> title mapping (assuming columns: item, title)"""
    titles = {}
    with open(title_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = str(row["item"])
            titles[item_id] = row["title"]
    return titles

def find_image_for_item(item_id: str, img_dir: str) -> Optional[str]:
    """Find corresponding image file based on item_id"""
    for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        path = os.path.join(img_dir, f"{item_id}{ext}")
        if os.path.exists(path):
            return path
    return None


def build_reasoning_prompt(user_id: str, hist_items: List[dict]) -> List[dict]:
    content = []
    content.append({
        "type": "text",
        "text": (
            f"User id: {user_id}.\n"
            f"Below are the user's most recent {len(hist_items)} interactions "
            f"with movies (images + titles) in chronological order."
        ),
    })

    for idx, item in enumerate(hist_items, start=1):
        content.append({"type": "image", "image": item["image_path"]})
        content.append({
            "type": "text",
            "text": f"History item {idx}: {item['title']} (id={item['item_id']}).",
        })

    content.append({
        "type": "text",
        "text": (
            "You are an expert in movie recommendation analysis. "
            "Use the Chain-of-Thought format:\n"
            "Step 1: Extract key multimodal attributes from the history items (use both image content and titles), "
            "including genre, visual style, color tone, scenes, actors, or thematic elements.\n"
            "Step 2: Infer the user's stable preference patterns.\n"
            "Step 3: Based on these patterns, propose 1–3 plausible next movies (or describe the characteristics of the most likely next movie) "
            "and explain why they match the user's interests. Do not assume the ground-truth next item is known and do not reference any final/last item.\n"
            "Step 4: Provide a brief conclusion summarizing the reasoning and the predicted preference profile.\n"
            "Write 4–6 sentences in English. Your explanation must explicitly cite cues from earlier interactions to justify the prediction."
        ),
    })

    return [{"role": "user", "content": content}]


# ================== Main Pipeline ==================
def main():
# 1. Load model & processor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # 2. Load titles map
    titles = load_titles(TITLE_FILE)

    results = []

    # 3. Iterate train.tsv
    with open(TRAIN_TSV, encoding="utf-8") as f:
        for line_idx, line in tqdm(
            enumerate(f), desc="Processing train.tsv", unit="line"
        ):
            line = line.strip()
            if not line:
                continue

            # Each line: user_id \t item1 item2 ... itemN
            try:
                user_id, seq_str = line.split("\t")
            except ValueError:
                # Skip malformed line
                continue

            all_items = seq_str.strip().split()
            if len(all_items) < K:
                continue

            # Last K interactions (in time order)
            recent_seq = all_items[-K:]
            if len(recent_seq) < 2:
                continue

            target_id = recent_seq[-1]   # Last one is target
            hist_ids = recent_seq[:-1]   # Previous are history

            # 4. Prepare history items
            hist_items = []
            skip_sample = False
            for iid in hist_ids:
                img_path = find_image_for_item(iid, IMG_DIR)
                if img_path is None:
                    # If image missing you can skip or ignore; here we skip
                    skip_sample = True
                    break
                title = titles.get(iid, "Unknown Title")
                hist_items.append(
                    {"item_id": iid, "title": title, "image_path": img_path}
                )

            # 5. Prepare target item
            target_img_path = find_image_for_item(target_id, IMG_DIR)
            if target_img_path is None:
                skip_sample = True

            if skip_sample:
                continue

            target_item = {
                "item_id": target_id,
                "title": titles.get(target_id, "Unknown Title"),
                "image_path": target_img_path,
            }

            # 6. Build prompt & inference
            messages = build_reasoning_prompt(user_id, hist_items, target_item)

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            # Move to device
            inputs = {
                k: (v.to(model.device) if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()
            }

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=256,
                )

            # Remove prompt tokens to keep only output
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            reasoning = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            results.append(
                {
                    "line_idx": line_idx,
                    "user_id": user_id,
                    "target_item": target_item,
                    "history_items": hist_items,
                    "reasoning": reasoning,
                }
            )

    # 7. Save results
    os.makedirs(os.path.dirname(SAVE_JSON), exist_ok=True)
    with open(SAVE_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Saved {len(results)} samples to {SAVE_JSON}.")


if __name__ == "__main__":
    main()