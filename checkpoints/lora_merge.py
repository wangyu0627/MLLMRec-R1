import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================
# Path Configuration
# Modify the variables below when switching dataset or experiment
# ==========================
ROOT       = "/root/autodl-tmp/MLLMRec-R1"       # Project root directory
DATASET    = "movielens"                       # Dataset name
EXP_NAME   = "qwen3_4b_sft_cot_int5/checkpoint-900"               # LoRA experiment/checkpoint folder
MERGED_NAME = "Qwen3-4B-COT-SFT-INT5"                    # Output folder name for merged model

# Construct absolute paths
BASE_MODEL = f"{ROOT}/Qwen3-4B"                             # Base model directory
LORA_DIR   = f"{ROOT}/checkpoints/{DATASET}/{EXP_NAME}"     # LoRA checkpoint directory
MERGED_OUT = f"{ROOT}/checkpoints/{DATASET}/{MERGED_NAME}"  # Output path for final merged model

print("Base Model Path :", BASE_MODEL)
print("LoRA Path       :", LORA_DIR)
print("Merged Output   :", MERGED_OUT)

# ==========================
# Load base model
# ==========================
# Load the original model weights before applying LoRA
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto"
)

# ==========================
# Load LoRA and attach to the base model
# ==========================
# LoRA checkpoint must contain adapter_config.json and adapter_model.bin
model = PeftModel.from_pretrained(
    base_model,
    LORA_DIR
)

# ==========================
# Merge LoRA weights into the base model
# ==========================
# merge_and_unload() merges adapter weights into base weights permanently
model = model.merge_and_unload()

# ==========================
# Save the merged full model
# ==========================
# The output directory (MERGED_OUT) will contain the final standalone model
model.save_pretrained(MERGED_OUT)

# Save tokenizer (usually from base model)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
tokenizer.save_pretrained(MERGED_OUT)

print("Finished: merged model saved to ->", MERGED_OUT)