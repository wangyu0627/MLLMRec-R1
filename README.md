# MLLMRec-R1
MLLMRec-R1: Incentivizing Reasoning Capability in Large Language Models for Multimodal Sequential Recommendation

## 🧰 Environment Setup for MLLMRec-R1

You can run the following command to download the codes faster:

```bash
conda create -y -n mllmrec python=3.11
conda activate mllmrec
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.57.3 trl==0.26.2 tqdm==4.67.1 pandas==2.3.3 peft==0.18.0 accelerate==1.12.0
```

## 💡 Dataset Structure and Download

You can download processed data files in the following datasets:

**Movielens**/ **Microlens**/ **Netflix** [[Google Drive]](https://drive.google.com/file/d/13cjTpuRXUHDi7aw76i6fUWmmeNISZZWG/view)

```plaintext
data/
├── microlens/                      # Same structure as MovieLens
├── movielens/
│   ├── images/                     # Item images (optional, used for MSR)
│   ├── processed/                  # sample
│   │   ├── test_generate_seed42_neg9.json
│   │   ├── train_generate_seed42_neg9_cot_0.0.json
│   │   └── train_generate_seed42_neg9_cot_0.1.json
│   ├── movielens_deepseek_cot.json # Multimodal CoT
│   │                               
│   ├── movielens_pairs.csv         # Preference pairs (positive/negative, pairwise)
│   ├── movielens_titles.csv        # Item metadata (item_id → title, etc.)
│   ├── train.tsv                   # Raw training interactions
│   └── test.tsv                    # Raw test interactions
└── netflix/                        # Same structure as MovieLens
```

## Download Backbone Model

Before running **Step 1 (SFT)** and subsequent steps, you must first download the **Qwen3-4B** and place it under the following directory:
```plaintext
/absolute_path/MLLMRec-R1/Qwen3-4B/
```

# 🚀 Examples to run the codes
## Step 1: Supervised Fine-Tuning (SFT)
> **Note:** `--root` must be the **absolute path** to the project directory (i.e., the folder that contains `train/`, `data/`, etc.).
### MovieLens
```bash
python train/sft.py --root /absolute_path/MLLMRec-R1 --backbone Qwen3-4B --use_cot --cot_prob 0.1 --dataset movielens --min_inter 10 --tag qwen3_4b_sft_cot
```
### Microlens
```bash
python train/sft.py --root /absolute_path/MLLMRec-R1 --backbone Qwen3-4B --use_cot --cot_prob 0.1 --epochs 5 --dataset microlens --min_inter 10 --tag qwen3_4b_sft_cot
```
### Netflix
```bash
python train/sft.py --root /absolute_path/MLLMRec-R1 --backbone Qwen3-4B --use_cot --cot_prob 0.1 --dataset netflix --min_inter 5 --tag qwen3_4b_sft_cot
```

## Step 2: Merge LoRA into the Base Model (GPU 0)
After SFT, the LoRA checkpoints are saved under:
```plaintext
checkpoints/<dataset>/qwen3_4b_sft_cot
where <dataset> is one of: movielens, microlens, netflix.
```
2.1 Set parameters in **checkpoints/lora_merge.py**

Example (MovieLens):
```bash
ROOT = "/absolute_path/MLLMRec-R1"
DATASET = "movielens"
EXP_NAME = "qwen3_4b_sft_cot" # your SFT result folder under checkpoints/movielens/
MERGED_NAME = "Qwen3-4B-COT-SFT"
```

2.2 Run LoRA merging on GPU 0

Use CUDA_VISIBLE_DEVICES=0 to force the merge to run on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python lora_merge.py
```

After merging, the final merged checkpoint will be saved under:

```plaintext
checkpoints/<dataset>/<MERGED_NAME>/
```

## Step 3: GRPO Post-Training (Reinforcement Fine-Tuning)
Run GRPO to further align the model with preference/reward signals. Make sure `--sft_tag` matches the merged checkpoint folder name produced in Step 2 (e.g., `Qwen3-4B-COT-SFT`).
### MovieLens
```bash
python train/grpo.py --dataset movielens --use_cot --cot_prob 0.05 --epochs 3 --root /absolute_path/MLLMRec-R1 --min_inter 10 --sft_tag Qwen3-4B-COT-SFT --tag qwen3_4b_grpo_cot --save_steps 500 --num_generations 8 --beta 0.1 
```
### Microlens
```bash
python train/grpo.py --dataset microlens --use_cot --cot_prob 0.1 --epochs 3 --root /absolute_path/MLLMRec-R1 --min_inter 10 --sft_tag Qwen3-4B-COT-SFT --tag qwen3_4b_grpo_cot --save_steps 500 --num_generations 8 --beta 0.05
```
### Netflix
```bash
python train/grpo.py --dataset netflix --use_cot --cot_prob 0.05 --epochs 2 --root /absolute_path/MLLMRec-R1 --min_inter 5 --sft_tag Qwen3-4B-COT-SFT --tag qwen3_4b_grpo_cot --save_steps 500 --num_generations 8 --beta 0.2
```

## Step 4: Evaluation (Distributed Inference)
Before running evaluation, **explicitly specify the GPUs to use** by setting `CUDA_VISIBLE_DEVICES`. This ensures the evaluation only uses the intended devices and matches `--nproc_per_node`.
```bash
export CUDA_VISIBLE_DEVICES=0,1
```
Evaluate the GRPO-trained model using distributed inference. Ensure that `--tag` points to the specific GRPO checkpoint to be evaluated. After GRPO, the LoRA checkpoints are saved under:
```plaintext
checkpoints/<dataset>/qwen3_4b_sft_grpo
```

### MovieLens/Microlens/Netflix
To switch datasets, simply change the value of --dataset (e.g., movielens, microlens, netflix).
When running multiple distributed jobs concurrently, change the port in --rdzv_endpoint (e.g., 29601, 29602, 29603, ...) to avoid port conflicts.
```bash
torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:29601 train/inference.py --root /absolute_path/MLLMRec-R1 --dataset movielens --min_inter 10 --top_k 3 --sft_tag Qwen3-4B-COT-SFT --tag qwen3_4b_grpo_cot --distributed
```







