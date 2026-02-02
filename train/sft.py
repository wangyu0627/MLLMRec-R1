import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.data_loader import RecSFTDataset
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from peft import LoraConfig

def parse_args():
    parser = argparse.ArgumentParser(description="SFT Training for Rec System")

    parser.add_argument("--dataset", type=str, required=True,help="Dataset name")
    parser.add_argument("--root", type=str, default="/root/autodl-tmp/MLLMRec-R1", help="Dataset root path")
    parser.add_argument("--backbone", type=str, default="Qwen3-4B", help="backbone")
    parser.add_argument("--use_cot", action="store_true", help="Enable Chain-of-Thought (CoT) reasoning in SFT prompts")
    parser.add_argument("--cot_prob", type=float, default=0.1, help="cot_prob")
    parser.add_argument("--min_inter", type=int, default=10, help="user sequence length")
    parser.add_argument("--num_neg", type=int, default=9, help="negative samples per user")
    parser.add_argument("--tag", type=str, default=None, help="LoRA adapter dir; if None, use ROOT/checkpoints/DATASET/qwen3_4b_sft_nocap")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accum", type=int, default=8)

    return parser.parse_args()


def main():
    args = parse_args()

    ROOT = args.root
    DATASET = args.dataset
    NUM_NEG = args.num_neg
    NUM_INTER = args.min_inter
    RANK = args.rank
    TAG = args.tag
    BACKBONE = args.backbone
    COT = args.use_cot
    COT_PROB = args.cot_prob

    model_name_or_path = f"{ROOT}/{BACKBONE}"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    model.train()

    # LoRA
    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    #
    train_ds = RecSFTDataset(
        root_path=ROOT,
        dataset_name=DATASET,
        split="train",
        num_neg=NUM_NEG,
        min_interactions=NUM_INTER,
        mode="train",
        seed=42,
        use_cot=COT,
        cot_prob=COT_PROB
    )

    print("Example sample:")
    print(train_ds[0]["messages"])

    #
    hf_train_ds = Dataset.from_list([train_ds[i] for i in range(len(train_ds))])

    output_dir = f"{ROOT}/checkpoints/{DATASET}/{TAG}"

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=20,
        save_steps=100,
        save_total_limit=20,
        bf16=True,
        completion_only_loss=False,
        dataset_text_field="messages",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=hf_train_ds,
        args=sft_config,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
