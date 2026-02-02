# grpo_train.py
import os
import re
import argparse
from typing import List
import torch
from datasets import Dataset
from peft import LoraConfig  # use a new LoRA adapter on top of the SFT model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig

# Custom dataset loader: mode="generate" should output
#   {"prompt": str, "target_id": str, "candidates": List[str], ...}
from utils.data_loader import RecGRPODataset


def parse_args():
    """Argument parser for GRPO trainer."""
    p = argparse.ArgumentParser("GRPO training for Recommendation System")

    # Dataset settings
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--root", type=str, default="/root/autodl-tmp/RL4MSRec")
    p.add_argument("--min_inter", type=int, default=10)
    p.add_argument("--num_neg", type=int, default=9)
    p.add_argument("--seed", type=int, default=42)

    # Backbone model tag used as initialization
    # The SFT model dir is ROOT/checkpoints/DATASET/sft_tag (e.g. Qwen3-4B-SFT)
    p.add_argument(
        "--sft_tag",
        type=str,
        required=True,
        help="SFT model tag used as initialization, e.g. Qwen3-4B-SFT",
    )
    p.add_argument("--use_cot", action="store_true", help="Enable Chain-of-Thought (CoT) reasoning in SFT prompts")
    p.add_argument("--cot_prob", type=float, default=0.1, help="cot_prob")

    # Training hyper-parameters
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=100)

    # GRPO-specific configs
    p.add_argument("--num_generations", type=int, default=4)   # generations per prompt
    p.add_argument("--max_completion_length", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--beta", type=float, default=0.1)  # KL coefficient (TRL default = 0.0)

    # Output LoRA tag after GRPO (new RL LoRA adapter)
    p.add_argument("--tag", type=str, default="qwen3_4b_grpo_lora")

    return p.parse_args()


ITEM_PATTERN = re.compile(r"^\[ITEM_(\d+)\]\s+(.+)$")

def rec_reward_func(completions, target_id, candidates, **kwargs):
    """
    Reward function for GRPO-based movie recommendation.
    Evaluation order:
        1) Check output format validity: must match [ITEM_xxxx] Movie Title
        2) If valid format → reward depends on hit or miss
        3) If format invalid → strong penalty
    """

    rewards = []

    for comp, tid in zip(completions, target_id):

        # Remove leading/trailing whitespace
        text = comp.strip()

        # If contains "</think>", keep only the part before it (ignore explanation)
        if "</think>" in text:
            text = text.split("</think>")[0].strip()
            # print(text)

        # 1) Format check using ITEM_PATTERN
        # Must start with: [ITEM_xxxx] Title
        m = ITEM_PATTERN.match(text)
        if not m:
            # Invalid format → strong negative reward
            rewards.append(-1.0)
            continue

        pred_id = m.group(1)  # Extract item id

        # 2) Reward based on accuracy
        if pred_id == tid:
            rewards.append(1.0)   # Correct format + correct item → highest reward
        else:
            rewards.append(0.3)   # Correct format but wrong item → small positive reward

    return rewards


def main():
    args = parse_args()

    # 0) Check GRPO batch divisibility:
    #    Effective Batch = world_size * batch_size * gradient_accumulation
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_bsz = world_size * args.batch * args.accum
    if effective_bsz % args.num_generations != 0:
        raise ValueError(
            f"Effective batch size (WORLD_SIZE*batch*accum)={effective_bsz} "
            f"must be divisible by num_generations={args.num_generations}."
        )

    # Backbone model is the SFT-finetuned full model:
    #   ROOT/checkpoints/DATASET/sft_tag  (e.g., Qwen3-4B-SFT)
    BASE_MODEL = f"{args.root}/checkpoints/{args.dataset}/{args.sft_tag}"

    # 1) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    # GRPO requires left padding + a valid pad_token
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) Load backbone model (SFT model)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.config.use_cache = False

    # Use the SFT model as backbone; GRPO will attach a new LoRA adapter via peft_config
    model = base_model
    model.train()

    # 3) Build GRPO training dataset (train split only)
    #    mode="generate" should output fields:
    #       "prompt": user history and candidate list description
    #       "target_id": ground-truth target item id (str)
    #       "candidates": list of candidate ids (List[str])
    train_ds = RecGRPODataset(
        root_path=args.root,
        dataset_name=args.dataset,
        split="train",
        num_neg=args.num_neg,
        min_interactions=args.min_inter,
        seed=args.seed,
        rec_top_k=1,              # GRPO usually optimizes 1 item
        add_format_instruction=True,  # strongly recommended
        use_cot=args.use_cot,
        cot_prob=args.cot_prob,
    )

    hf_train_ds = Dataset.from_list([train_ds[i] for i in range(len(train_ds))])
    print(hf_train_ds[0])

    # New GRPO LoRA adapter will be saved here
    output_dir = f"{args.root}/checkpoints/{args.dataset}/{args.tag}"

    # 4) LoRA configuration for GRPO (new RL adapter on top of SFT model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",  # let PEFT apply LoRA to all linear layers
    )

    # 5) GRPO configuration
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=True,

        # We must keep all dataset fields for the reward function
        remove_unused_columns=False,

        # GRPO generation configs
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
    )

    # 6) Construct GRPO trainer (model + new LoRA)
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=hf_train_ds,
        reward_funcs=rec_reward_func,
        processing_class=tokenizer,
        peft_config=lora_config,  # attach a new LoRA adapter for RL
    )

    # 7) Train and save
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[DONE] GRPO training finished. Saved to: {output_dir}")


if __name__ == "__main__":
    main()
