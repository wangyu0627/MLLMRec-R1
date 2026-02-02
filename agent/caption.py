import os
import csv
import json
import torch
from tqdm import tqdm
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor


DATASET_NAME = "microlens"
root = "/root/autodl-tmp/MLLMRec-R1"
model_id = f"{root}/Qwen3-VL-8B-Instruct"
IMG_DIR = f"{root}/data/{DATASET_NAME}/images"
TITLE_FILE = f"{root}/data/{DATASET_NAME}/{DATASET_NAME}_titles.csv"
SAVE_JSON = f"{root}/data/{DATASET_NAME}_captions.json"

print(f"\n=== Running Caption Generation for Dataset: {DATASET_NAME} ===")
print(f"Model:       {model_id}")
print(f"Images:      {IMG_DIR}")
print(f"Title File:  {TITLE_FILE}")
print(f"Output:      {SAVE_JSON}\n")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

titles = {}
with open(TITLE_FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        titles[str(row["item"])] = row["title"]

results = {}
for file in tqdm(sorted(os.listdir(IMG_DIR))):
    if not file.lower().endswith(("jpg","jpeg","png","webp","bmp")):
        continue

    item_id = os.path.splitext(file)[0]   # "1.jpg" -> "1"
    title = titles.get(item_id, "Unknown Title")

    img_path = os.path.join(IMG_DIR, file)
    user_prompt = (
        f"You are generating a caption for a movie/show dataset.\n"
        f"Title: '{title}'\n"
        f"Task: Write one caption (no more than 50 words) describing the image content.\n"
        f"The caption should be visually grounded while loosely consistent with the theme.\n"
        f"Do NOT rewrite the title. Do NOT mention the title explicitly.\n"
        f"Output only the caption sentence."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    )

    inputs = {
        k: (v.to(model.device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }

    generated_ids = model.generate(**inputs, do_sample=False,  max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    caption = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    results[item_id] = {"title": title, "caption": caption}
    # print(f"[{item_id}] {title} -> {caption}")


with open(SAVE_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nCompleted! Captions saved to {SAVE_JSON}, total {len(results)}.")
