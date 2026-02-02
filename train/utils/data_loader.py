# utils/data_loader.py
import os
import json
import random
import pandas as pd
from torch.utils.data import Dataset
import re

class InteractionDataset(Dataset):
    """
    Basic interaction dataset.

    This class reads a TSV file of the form:
        user_id \t item_seq

    and converts each row into a sample:
        {
            "user_id": <str>,
            "history_ids": [item_id_1, ..., item_id_n-1],
            "target_id": item_id_n,
        }

    Only users with at least `min_interactions` interactions are kept, and
    only the last `history_len + 1` items in the sequence are used.
    """

    def __init__(
        self,
        root_path: str,
        dataset_name: str = "movielens",
        split: str = "train",
        min_interactions: int = 10,
        history_len: int = 9,
    ):
        tsv_path = f"{root_path}/data/{dataset_name}/{split}.tsv"
        df = pd.read_csv(tsv_path, sep="\t", header=None, names=["user_id", "item_seq"])

        self.samples = []
        for _, row in df.iterrows():
            items = str(row["item_seq"]).split()
            if len(items) < min_interactions:
                continue
            # Keep the last (history_len + 1) items: history + target
            seq = items[-(history_len + 1):]
            self.samples.append(
                {
                    "user_id": str(row["user_id"]),
                    "history_ids": seq[:-1],
                    "target_id": seq[-1],
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class RecSFTDataset(Dataset):
    """
    SFT dataset for recommendation with fixed negative sampling and caching.

    Key design:
    - All negative items and candidate lists are sampled ONCE when the dataset is
      built for the first time (for a specific combination of
      dataset_name / split / mode / seed / num_neg).
    - The processed samples are saved to:
        ROOT/data/{dataset_name}/processed/{split}_{mode}_seed{seed}_neg{num_neg}.json
    - Subsequent runs with the same parameters will directly load from this file
      instead of resampling, ensuring strict reproducibility.

    Modes:
    - mode = "train":
        each sample is:
        {
            "messages": [
                {"role": "user", "content": <prompt>},
                {"role": "assistant", "content": <completion>}
            ]
        }

    - mode = "generate":
        each sample is:
        {
            "prompt": <prompt>,
            "target_id": <str>,
            "candidates": [<item_id_str>, ...]
        }
    """

    def __init__(
        self,
        root_path: str,
        dataset_name: str = "movielens",
        split: str = "train",
        num_neg: int = 9,
        min_interactions: int = 10,
        mode: str = "train",
        seed: int = 42,
        use_cot: bool = False,
        cot_prob: float = 0.1
    ):
        # Basic config
        self.root_path = root_path
        self.cot_path = f"{root_path}/data/{dataset_name}/{dataset_name}_deepseek_cot.json"
        self.dataset_name = dataset_name
        self.split = split
        self.num_neg = num_neg
        self.min_interactions = min_interactions
        history_len = min_interactions - 1
        self.history_len = history_len
        self.cot_prob =cot_prob
        self.mode = mode
        self.seed = seed

        # Directory to store processed (fixed) data
        self.processed_dir = f"{root_path}/data/{dataset_name}/processed"
        os.makedirs(self.processed_dir, exist_ok=True)

        if use_cot:
            # New cache when CoT is enabled
            self.cache_file = os.path.join(
                self.processed_dir,
                f"{split}_{mode}_seed{seed}_neg{num_neg}_cot_{cot_prob}_int_{min_interactions}.json",
            )
        else:
            # Original cache (fully backward compatible)
            self.cache_file = os.path.join(
                self.processed_dir,
                f"{split}_{mode}_seed{seed}_neg{num_neg}_int_{min_interactions}.json",
            )

        # If cache exists, load and return directly (no re-sampling)
        if os.path.exists(self.cache_file):
            print(f"[LOAD FIXED DATA] {self.cache_file}")
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self.samples = json.load(f)
            return

        print(f"[BUILD DATA FIRST TIME] split={split}, mode={mode}, seed={seed}, num_neg={num_neg}, cot={cot_prob}, int={min_interactions}")

        # Use a local random generator so that sampling is reproducible and isolated
        self.rng = random.Random(seed)

        # Base interaction dataset (user_id, history_ids, target_id)
        self.inter_ds = InteractionDataset(
            root_path=root_path,
            dataset_name=dataset_name,
            split=split,
            min_interactions=min_interactions,
            history_len=history_len,
        )

        # Load item titles
        titles_path = f"{root_path}/data/{dataset_name}/{dataset_name}_titles.csv"
        titles_df = pd.read_csv(titles_path)
        item_col = titles_df.columns[0]
        title_col = titles_df.columns[1]
        self.id2title = {
            str(row[item_col]): str(row[title_col]) for _, row in titles_df.iterrows()
        }

        # Load image captions
        # caption_path = f"{root_path}/data/{dataset_name}/{dataset_name}_captions.json"
        # with open(caption_path, "r", encoding="utf-8") as f:
        #     caption_data = json.load(f)
        # self.id2caption = {}
        # for item_id, info in caption_data.items():
        #     cap = info.get("caption", "")
        #     self.id2caption[str(item_id)] = str(cap)

        self.idx2reasoning = {}
        with open(self.cot_path, "r", encoding="utf-8") as f:
            cot_data = json.load(f)

        # cot format: list[{line_idx, reasoning, ...}]
        for row in cot_data:
            if "line_idx" in row:
                self.idx2reasoning[int(row["line_idx"])] = str(row.get("reasoning", ""))

        self.all_item_ids = list(self.id2title.keys())

        # Pre-build all examples ONCE so that negatives & candidates are fixed
        self.samples = []
        self._build_all_samples()

        # Save processed samples to JSON for future reuse
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.samples, f, ensure_ascii=False, indent=2)
        print(f"[SAVE FIXED DATA] {self.cache_file}")

    def _sample_negatives(self, history_ids, target_id):
        """
        Sample a fixed set of negative item IDs for one example.

        This method uses self.rng (a local Random instance with a fixed seed),
        so the sampling result is deterministic for a given dataset configuration.
        """
        forbidden = set(history_ids + [target_id])
        neg_ids = []
        while len(neg_ids) < self.num_neg:
            cand = self.rng.choice(self.all_item_ids)
            if cand not in forbidden:
                neg_ids.append(cand)
                forbidden.add(cand)
        return neg_ids

    def extract_step(self, reasoning: str) -> str:
        """
        Extract Step 3 (Retrieval criteria) from reasoning text.
        Return empty string if not found.
        """
        if not reasoning:
            return ""

        # Match "Step 3: ..." until "Step 4:" or end of text
        pattern = r"Step\s*3\s*:(.*?)(?:Step\s*4\s*:|$)"
        match = re.search(pattern, reasoning, flags=re.S | re.I)
        if not match:
            return ""

        step = match.group(1).strip()

        # Optional: remove leading labels like "Retrieval criteria:"
        step = re.sub(r"^Retrieval criteria\s*:\s*", "", step, flags=re.I)

        # Optional: compress whitespace
        step = re.sub(r"\s+", " ", step).strip()

        return step

    def _build_prompt(self, history_ids, candidates, reasoning):
        """
        Build the user prompt given history and candidate items.

        The history sequence and candidate list both contain item IDs, which are
        converted into human-readable titles and optional image captions.
        """
        lines = []
        lines.append("You are a movie recommendation assistant.")
        lines.append(
            "Here are the user's most recently watched movies sequence:"
        )
        for i, iid in enumerate(history_ids, 1):
            title = self.id2title.get(iid, "Unknown Title")
            # cap = self.id2caption.get(iid, "")
            text = title
            # text = f"{title} — {cap}" if cap else title
            lines.append(f"{i}. [ITEM_{iid}] {text}")

        lines.append("")
        lines.append("Candidate movies:")
        # Candidate list (already shuffled at construction time)
        for iid in candidates:
            title = self.id2title.get(iid, "Unknown Title")
            # cap = self.id2caption.get(iid, "")
            text = title
            # text = f"{title} — {cap}" if cap else title
            lines.append(f"[ITEM_{iid}] {text}")

        lines.append("")
        if reasoning and random.random() < self.cot_prob:
            # step = self.extract_step(reasoning)
            # if self.mode == "generate" and step is not None:
            lines.append("Possible reasoning process to help predict the next item, for reference ONLY.")
            lines.append("<think>")
            lines.append(reasoning)
            lines.append("</think>")

        lines.append("")
        lines.append(
            "Now answer the following question.\n"
            "Only output in format [ITEM_xxxx] Movie Title.\n"
            "No additional caption, summary, comments, or explanation."
        )

        prompt = "\n".join(lines)
        return prompt

    def _build_all_samples(self):
        """
        Pre-compute negatives, candidates, prompt, and completion for all examples.

        The result is stored in self.samples, which will be serialized to disk
        after construction. This guarantees that the dataset is fixed for a
        given configuration.
        """
        for sample_idx, rec in enumerate(self.inter_ds):
            history_ids = rec["history_ids"]
            target_id = rec["target_id"]

            # 1) Sample fixed negatives for this example
            neg_ids = self._sample_negatives(history_ids, target_id)

            # 2) Build candidate list: 1 positive + N negatives, then shuffle ONCE
            candidates = [str(target_id)] + [str(nid) for nid in neg_ids]
            self.rng.shuffle(candidates)

            # 3) Build prompt
            reasoning = self.idx2reasoning.get(sample_idx, "")
            prompt = self._build_prompt(history_ids, candidates, reasoning)

            # 4) Build ground-truth completion
            target_title = self.id2title.get(target_id, "Unknown Title")
            completion = f"[ITEM_{target_id}] {target_title}"

            if self.mode == "train":
                # Conversational language modeling format for SFT
                messages = [
                    {
                        "role": "user",
                        "content": prompt,
                    },
                    {
                        "role": "assistant",
                        "content": completion,
                    },
                ]
                self.samples.append({"messages": messages})

            elif self.mode == "generate":
                # For ranking / evaluation / generation-based inference
                self.samples.append(
                    {
                        "prompt": prompt,
                        "target_id": str(target_id),
                        "candidates": [str(cid) for cid in candidates],
                    }
                )

            else:
                raise ValueError(f"Unsupported mode: {self.mode}")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Simply return pre-built samples; no randomness here.
        return self.samples[idx]

class RecGRPODataset(Dataset):
    """
    Dataset wrapper for GRPO training in recommendation setting.

    This class reuses RecSFTDataset with mode="generate" to obtain:
        - prompt: user interaction history + candidate list description
        - target_id: ground-truth item id (positive sample)
        - candidates: list[str], candidate item ids (top-K)

    Then it optionally appends a formatting instruction to the prompt,
    so that the model is explicitly guided to output recommendations in
    the desired [ITEM_xxxx] Movie Title format.

    The resulting samples are dictionaries:
        {
            "prompt": str,
            "target_id": str,
            "candidates": List[str],
            ... (other fields from RecSFTDataset if needed)
        }

    These fields are exactly what GRPOTrainer needs when using a custom
    reward function (rec_reward_func).
    """

    def __init__(
        self,
        root_path: str,
        dataset_name: str,
        split: str = "train",
        num_neg: int = 9,
        min_interactions: int = 10,
        seed: int = 42,
        # GRPO / prompt-related arguments
        rec_top_k: int = 1,
        add_format_instruction: bool = True,
        instruction_template: str | None = None,
        use_cot: bool = False,
        cot_prob: float = 0.1,
    ) -> None:
        """
        Args:
            root_path: project root path, same as in RecSFTDataset.
            dataset_name: dataset name, e.g., "microlens".
            split: "train", "valid", or "test". For GRPO, usually "train".
            num_neg: number of negatives used when building generate-style samples.
            min_interactions: minimum user sequence length.
            seed: random seed for sampling negatives / shuffling.

            rec_top_k: number of items to recommend in the instruction.
                       This is only used to build the text instruction,
                       it does NOT change the candidate list size.
            add_format_instruction: whether to append a strict format hint
                                    to the prompt (recommended = True).
            instruction_template: optional custom template. If None, use
                                  a default single-line format instruction.
        """
        super().__init__()

        # 1) Underlying dataset: use RecSFTDataset in "generate" mode
        #    so that it outputs prompt/target_id/candidates for each example.
        base_ds = RecSFTDataset(
            root_path=root_path,
            dataset_name=dataset_name,
            split=split,
            num_neg=num_neg,
            min_interactions=min_interactions,
            mode="generate",
            seed=seed,
            use_cot=use_cot,
            cot_prob=cot_prob,
        )

        self.samples = []
        self.rec_top_k = rec_top_k
        self.add_format_instruction = add_format_instruction

        # 2) Prepare default instruction template if not provided
        #    This is similar to what you use in inference.py, but adapted
        #    for single-line GRPO style (or you can keep multi-line).
        if instruction_template is None:
            instruction_template = (
                "\n\n[OUTPUT GUIDELINES]\n"
                "You MUST answer with only one movie from candidate movies.\n"
                "The output token MUST be in the form:\n"
                "[ITEM_xxxx] Movie Title\n\n"
                "If you want to provide reasoning, you may append it ONLY after answer\n"
                "in a single line using:\n"
                "<think> your reasoning here </think>\n\n"
            )

        self.instruction_template = instruction_template

        # 3) Build GRPO-ready samples
        for ex in base_ds:
            prompt = ex["prompt"]
            if add_format_instruction:
                prompt += instruction_template

            self.samples.append({
                "prompt": prompt,
                "target_id": ex["target_id"],
                "candidates": ex["candidates"],
            })

    def __len__(self):
            return len(self.samples)

    def __getitem__(self, idx):
            return self.samples[idx]