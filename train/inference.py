import argparse
import math
import os
import re
import json
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from utils.data_loader import RecSFTDataset
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Top-K generation eval for RecSFT model")

    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. microlens / netflix / movielens ...")
    parser.add_argument("--root", type=str, default="/root/autodl-tmp/MLLMRec-R1", help="Project root path")
    parser.add_argument("--backbone", type=str, default="Qwen3-4B", help="Backbone")
    parser.add_argument("--top_k", type=int, default=3, help="Top-K for HR@K and NDCG@K")
    parser.add_argument("--min_inter", type=int, default=10, help="user sequence length")
    parser.add_argument("--num_neg", type=int, default=9, help="negative samples per user for RecSFTDataset (test)")
    parser.add_argument("--device", type=str, default="cuda", help="device for inference, e.g. cuda / cpu (non-distributed only)")
    parser.add_argument("--sft_tag", type=str, default=None, help="SFT model tag used as initialization, e.g. Qwen3-4B-SFT")
    parser.add_argument("--tag", type=str, default=None, help="LoRA adapter dir name under ROOT/checkpoints/DATASET/")
    parser.add_argument("--distributed", action="store_true", help="Use torchrun multi-GPU data-parallel inference")

    return parser.parse_args()


def build_user_messages(prompt: str):
    """
    Build messages in the same format as messages[0] during SFT training,
    and reuse this format at inference time.
    """
    return [{"role": "user", "content": prompt}]


def extract_item_id(text: str):
    """
    Extract an item ID of the form ITEM_xxxx from the generated text.

    As long as the model output contains '[ITEM_1234]' or 'ITEM 1234'
    (with optional '_', space, or '-' after 'ITEM'), we treat it as a valid ID.
    """
    m = re.search(r"ITEM[_\s-]?(\d+)", text)
    return m.group(1) if m else None


def generate_topk_items(
    model,
    tokenizer,
    prompt: str,
    top_k: int,
    device: str,
):
    """
    Use free-form generation to obtain Top-K recommended item IDs.

    The model is prompted to output K lines in the format:
        [ITEM_xxx]
        [ITEM_xxx]
        ...
    Then we parse all 'ITEM_xxx' occurrences and keep the first K.
    """
    user_messages = build_user_messages(
        prompt
        # + f"\nMUST RECOMMEND {top_k} ITEMS!!! NOT ONE! Output format strictly:\n[ITEM_xxx]\n[ITEM_xxx]\n..."
        + f"\nPlease recommend {top_k} items, output format strictly:\n[ITEM_xxx]\n[ITEM_xxx]\n..."
    )
    prompt_text = tokenizer.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        top_p=0.7,
        temperature=0.1,
        # pad_token_id=tokenizer.eos_token_id,
    )

    gen = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    ids = re.findall(r"ITEM[_\s-]?(\d+)", gen)
    return ids[:top_k]  # Force to keep only the first K IDs


def _init_distributed_if_needed(args):
    """
    Initialize torch.distributed if:
    - args.distributed is True
    - torchrun env vars are present
    """
    distributed = bool(args.distributed) and ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)

    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
        return distributed, rank, world_size, local_rank, device

    return False, 0, 1, 0, args.device


def main():
    args = parse_args()

    distributed, rank, world_size, local_rank, DEVICE = _init_distributed_if_needed(args)

    ROOT = args.root
    DATASET = args.dataset
    TOP_K = args.top_k
    NUM_INTER = args.min_inter
    NUM_NEG = args.num_neg
    SFT_TAG = args.sft_tag
    TAG = args.tag
    BACKBONE = args.backbone

    if TAG is None:
        raise ValueError("--tag must be provided (LoRA adapter dir name).")

    if SFT_TAG is None:
        BASE_MODEL = f"{ROOT}/{BACKBONE}"
    else:
        BASE_MODEL = f"{ROOT}/checkpoints/{DATASET}/{SFT_TAG}"

    ADAPTER_DIR = f"{ROOT}/checkpoints/{DATASET}/{TAG}"

    # ---- Load tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load base model ----
    # IMPORTANT:
    # - For distributed data-parallel inference: each process loads model on its local GPU.
    # - Avoid mixing device_map="auto" with model.to(device) in this mode.
    if distributed:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(DEVICE)
    else:
        # keep original behavior (but fix dtype arg name)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    # ---- Load LoRA adapter on top of the base model ----
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    if not distributed:
        model.to(DEVICE)  # keep your original behavior

    # === Build test dataset ===
    test_ds = RecSFTDataset(
        root_path=ROOT,
        dataset_name=DATASET,
        split="test",
        num_neg=NUM_NEG,
        min_interactions=NUM_INTER,
        mode="generate",
        seed=42,
    )

    # Local metrics / results (per-rank)
    total_local = 0
    hit_k_local = 0
    ndcg_k_sum_local = 0.0
    results_local = []

    # Shard indices by rank
    indices = list(range(rank, len(test_ds), world_size))

    iterator = indices
    if (not distributed) or rank == 0:
        iterator = tqdm(indices, desc=f"Gen-TopK eval (world_size={world_size})")

    for idx in iterator:
        ex = test_ds[idx]
        prompt = ex["prompt"]
        target_id = ex["target_id"]  # e.g. "1129"

        gen_ids = generate_topk_items(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            top_k=TOP_K,
            device=DEVICE,
        )

        total_local += 1

        # Optional: print first few cases on rank0 only
        if ((not distributed) or rank == 0) and total_local <= 5:
            print("\n=== Case", total_local, "===")
            print("Target ID :", target_id)
            print("Gen Top-K :", gen_ids)

        # If empty generation -> miss
        if len(gen_ids) == 0:
            results_local.append({
                "index": None,  # global index set later (rank0 can reindex)
                "target_id": target_id,
                "pred_topk": gen_ids,
                "hit": 0,
                "rank": None,
            })
            continue

        hit = int(target_id in gen_ids)
        if hit:
            hit_k_local += 1
            r = gen_ids.index(target_id) + 1
            ndcg_k_sum_local += 1.0 / math.log2(r + 1)
            rank_pos = r
        else:
            rank_pos = None

        results_local.append({
            "index": None,
            "target_id": target_id,
            "pred_topk": gen_ids,
            "hit": hit,
            "rank": rank_pos,
        })

    # ---- Gather results + reduce metrics ----
    if distributed:
        # Gather Python objects (results) from all ranks
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, results_local)
        results = [x for part in gathered for x in part]

        # Reduce scalar metrics
        metrics = torch.tensor(
            [total_local, hit_k_local, ndcg_k_sum_local],
            device=DEVICE,
            dtype=torch.float64
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total = int(metrics[0].item())
        hit_k = int(metrics[1].item())
        ndcg_k_sum = float(metrics[2].item())
    else:
        results = results_local
        total = total_local
        hit_k = hit_k_local
        ndcg_k_sum = ndcg_k_sum_local

    # ---- Only rank0 saves / prints ----
    if (not distributed) or rank == 0:
        hr_k = hit_k / total if total > 0 else 0.0
        ndcg_k = ndcg_k_sum / total if total > 0 else 0.0

        print(f"\nTotal test: {total}")
        print(f"Gen HR@{TOP_K}   : {hr_k:.4f}")
        print(f"Gen NDCG@{TOP_K} : {ndcg_k:.4f}")

        # Reindex for readability
        for i, r in enumerate(results, 1):
            r["index"] = i

        # Save
        result_dir = f"{ROOT}/result/{DATASET}"
        os.makedirs(result_dir, exist_ok=True)

        save_path = os.path.join(result_dir, f"{TAG}_{DATASET}_top{TOP_K}.csv")
        df = pd.DataFrame(results)
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"\n[Saved Results] {save_path}")

        summary = {
            "dataset": DATASET,
            "top_k": TOP_K,
            "HR@K": hr_k,
            "NDCG@K": ndcg_k,
            "total_test": total,
            "distributed": bool(distributed),
            "world_size": world_size,
        }

        summary_path = os.path.join(result_dir, f"{TAG}_{DATASET}_top{TOP_K}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[Saved Metrics] {summary_path}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()