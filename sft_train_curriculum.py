import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed,
    AutoConfig
)
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import pandas as pd
from datasets import Dataset
from typing import Any, Dict, List, Optional

from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import random
import numpy as np
import multiprocessing as mp
from transformers import TrainerCallback
from peft import PeftModel
import argparse
# Set seed for reproducibility
set_seed(42)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("WANDB_PROJECT", "en-es")


parser = argparse.ArgumentParser(description="Fine-tune SmolLM2 EN→ES")

parser.add_argument("--num_frozen_layers", type=int, default=28,
                    help="Number of frozen layers / total hidden layers to use")
parser.add_argument("--num_task_layers", type=int, default=2,
                    help="Number of task-specific layers")
parser.add_argument("--lora", action="store_true",
                    help="Enable LoRA fine-tuning")
parser.add_argument("--no-lora", dest="lora", action="store_false",
                    help="Disable LoRA fine-tuning")
parser.set_defaults(lora=False)

parser.add_argument("--qlora", action="store_true",
                    help="Enable QLoRA fine-tuning (not used in current script)")
parser.add_argument("--no-qlora", dest="qlora", action="store_false",
                    help="Disable QLoRA fine-tuning")
parser.set_defaults(qlora=False)

parser.add_argument("--bf16", action="store_true",
                    help="Use bfloat16")
parser.add_argument("--no-bf16", dest="bf16", action="store_false",
                    help="Do not use bfloat16")
parser.set_defaults(bf16=True)

parser.add_argument("--aug_size", type=float, default=1.0,
                    help="Fraction of synthetic data size vs original")
parser.add_argument("--max_train_sample_size", type=int, default=100_000,
                    help="Maximum number of training samples")
parser.add_argument("--max_val_sample_size", type=int, default=10_000,
                    help="Maximum number of validation samples")
# epoch
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
# weight decay
parser.add_argument("--weight_decay", type=float, default=0.00, help="Weight decay for optimizer")

# usage: sft_train.py --num_frozen_layers 28 --num_task_layers 2 --lora --bf16 --aug_size 1.0 --max_train_sample_size 100000 --max_val_sample_size 10000

args = parser.parse_args()

num_frozen_layers = args.num_frozen_layers
num_task_layers = args.num_task_layers
lora = args.lora
qlora = args.qlora
bf16 = args.bf16
aug_size = args.aug_size
max_train_sample_size = args.max_train_sample_size
max_val_sample_size = args.max_val_sample_size

method = None
if lora:
    method = "LoRA"
    if qlora:
        method = "QLoRA"
else:
    method = "MultiTask"

RUN_NAME = f"{method}-F{num_frozen_layers}-T{num_task_layers}-aug{aug_size}-bf16{int(bf16)}"
print(f"RUN NAME: {RUN_NAME}")
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
DATASET_PATH = "exp-data/en-es-train-val.parquet"
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = args.epochs
OUTPUT_DIR = "./exp-data/models/" + RUN_NAME
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
WEIGHT_DECAY = args.weight_decay
GRAD_CLIP_VAL = 1.0
DATALOADER_WORKERS = mp.cpu_count() - 1 if mp.cpu_count() > 1 else 1

CURRICULUM_STAGES = [
    {"label": 0, "max_length": 64},
    {"label": 2, "max_length": 128},
    {"label": 4, "max_length": 256},
    {"label": 5, "max_length": 384},
    {"label": 7, "max_length": 480},
    {"label": 10, "max_length": None},  # None represents no upper bound (max)
]

new_config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
new_config.num_hidden_layers = num_frozen_layers + num_task_layers

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

peft_config = None
if lora:
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj",   # attention query projection
            "k_proj",   # attention key projection
            "v_proj",   # attention value projection
            "o_proj",   # attention output projection
            "gate_proj",  # MLP gate projection
            "up_proj",    # MLP up projection
            "down_proj",  # MLP down projection
        ],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        config=new_config
    )
else:
    if num_task_layers < 1:
        raise ValueError("num_task_layers must be at least 1 when not using LoRA.")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        config=new_config,
    )
    # Freeze all layers except the last `num_task_layer` in model.model.layers
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze the last `num_task_layer` transformer blocks
    for layer in model.model.layers[-num_task_layers:]:
        for param in layer.parameters():
            param.requires_grad = True
    # model.model.norm
    for param in model.model.norm.parameters():
        param.requires_grad = True
    # Unfreeze the LM head
    for param in model.lm_head.parameters():
        param.requires_grad = True


trainable_params = 0
non_trainable_params = 0

for _, param in model.named_parameters():
    num_params = param.numel()
    if param.requires_grad:
        trainable_params += num_params
    else:
        non_trainable_params += num_params

print(f"Trainable parameters: {trainable_params:,}")
print(f"Non-trainable parameters: {non_trainable_params:,}")
print(f"Total parameters: {(trainable_params + non_trainable_params):,}")

train_proc_df = pd.read_parquet("exp-data/en-es-train-val.parquet")
train_df = train_proc_df[train_proc_df["split"] == "train"]
train_df = train_df.sample(min(max_train_sample_size, len(train_df)), random_state=42).reset_index(drop=True)
val_df = train_proc_df[train_proc_df["split"] == "val"]
val_df = val_df.sample(min(max_val_sample_size, len(val_df)), random_state=42).reset_index(drop=True)
print(f"Training size before augmentation: {len(train_df)}")
print(f"Validation size before augmentation: {len(val_df)}")

if aug_size > 0:
    synthetic_df = pd.read_parquet("exp-data/synthetic-en-es-data.parquet")
    original_synthetic_size = len(synthetic_df)
    size = int(len(train_proc_df) * aug_size)
    size = min(size, len(synthetic_df))
    synthetic_df = synthetic_df.sample(n=size, random_state=42).reset_index(drop=True)
    print(f"Using {len(synthetic_df)} augmented samples out of {original_synthetic_size} available.")
    train_synthetic_df = synthetic_df[synthetic_df["split"] == "train"].reset_index(drop=True)
    val_synthetic_df = synthetic_df[synthetic_df["split"] == "val"].reset_index(drop=True)
    train_df = pd.concat([train_df, train_synthetic_df]).reset_index(drop=True)
    val_df = pd.concat([val_df, val_synthetic_df]).reset_index(drop=True)
    # new size
    print(f"New training size after augmentation: {len(train_df)}")
    print(f"New validation size after augmentation: {len(val_df)}")

# remove duplicates in train_df based on EN and ES columns
train_df = train_df.drop_duplicates(subset=["EN", "ES"]).reset_index(drop=True)
val_df = val_df.drop_duplicates(subset=["EN", "ES"]).reset_index(drop=True)

if "length" not in train_df.columns or "length" not in val_df.columns:
    raise ValueError("Both train and validation dataframes must contain a 'length' column for curriculum learning.")

def filter_by_length_range(
    df: pd.DataFrame, min_length: Optional[float], max_length: Optional[float]
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if min_length is not None:
        mask &= df["length"] > min_length
    if max_length is not None:
        mask &= df["length"] <= max_length
    return df[mask].reset_index(drop=True)

# Instruction template for translation
INSTRUCTION = "English: {en} Spanish:"

def formatting_prompts_func(example):
    text = INSTRUCTION.format(en=example["EN"]) + " " + example["ES"] + tokenizer.eos_token
    return {"text": text}

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )

def build_sft_dataset(df: pd.DataFrame) -> Optional[Dataset]:
    if df.empty:
        return None
    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset = dataset.map(formatting_prompts_func, remove_columns=dataset.column_names)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return dataset

def sample_examples(
    df: pd.DataFrame, fallback_df: pd.DataFrame, num_samples: int = 3
) -> List[Dict[str, Any]]:
    source_df = df if not df.empty else fallback_df
    if source_df.empty:
        return []
    sample_size = min(num_samples, len(source_df))
    return source_df.sample(n=sample_size, random_state=42).to_dict(orient="records")


class TranslationEvalCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        samples: List[Dict[str, Any]],
        instruction_template: str,
        stage_name: str,
        max_new_tokens: int = 128,
    ):
        self.tokenizer = tokenizer
        self.samples = samples
        self.instruction_template = instruction_template
        self.stage_name = stage_name
        self.max_new_tokens = max_new_tokens

    def generate_translation(self, model, prompt: str) -> str:
        if self.tokenizer.pad_token is None and self.tokenizer.pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=pad_token_id,
            )
        full_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if "Spanish:" in full_text:
            spanish = full_text.split("Spanish:", 1)[1].strip()
            eos_token = self.tokenizer.eos_token or ""
            if eos_token and eos_token in spanish:
                spanish = spanish.split(eos_token, 1)[0].strip()
        else:
            spanish = full_text
        return spanish.strip()

    def on_epoch_end(self, args, state, control, **kwargs):
        if not self.samples:
            return control

        model = kwargs["model"]
        was_training = model.training
        model.eval()

        print("\n" + "=" * 80)
        epoch_display = f"{state.epoch:.1f}" if state.epoch is not None else "?"
        print(f"END OF EPOCH {epoch_display} - {self.stage_name} SAMPLE TRANSLATIONS")
        print("=" * 80)

        for i, sample in enumerate(self.samples, 1):
            prompt = self.instruction_template.format(en=sample["EN"])
            current_pred = self.generate_translation(model, prompt)
            reference = sample["ES"]

            print(f"EN → {sample['EN']}")
            print(f"REF → {reference}")
            print(f"CURRENT (Fine-tuned) → {current_pred}")
            print("-" * 80)

        if was_training:
            model.train()

def build_stage_training_args(stage_name: str, stage_output_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=stage_output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=1000,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=False,
        bf16=bf16,
        report_to=[],
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit" if lora else "adamw_torch",
        remove_unused_columns=False,
        run_name=stage_name,
        weight_decay=WEIGHT_DECAY,
        save_total_limit=0,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=f"exp-data/runs/{RUN_NAME}/logs/{stage_name}",
        max_grad_norm=GRAD_CLIP_VAL,
        dataloader_num_workers=DATALOADER_WORKERS,
        logging_strategy="steps",
        disable_tqdm=False,
    )

final_trainer = None
current_model = model
previous_max_length = None

print("Starting fine-tuning for English → Spanish translation with curriculum learning...")

for stage_idx, stage in enumerate(CURRICULUM_STAGES):
    stage_min_length = previous_max_length
    stage_max_length = stage["max_length"]
    previous_max_length = stage_max_length

    stage_train_df = filter_by_length_range(train_df, stage_min_length, stage_max_length)
    stage_val_df = filter_by_length_range(val_df, stage_min_length, stage_max_length)

    if stage_train_df.empty:
        continue

    stage_label = stage["label"]
    stage_suffix = stage_max_length if stage_max_length is not None else "max"
    stage_name = f"{RUN_NAME}-stage{stage_label}-len{stage_suffix}"
    stage_output_dir = os.path.join(OUTPUT_DIR, f"stage-{stage_label}-len-{stage_suffix}")

    print(
        f"\nStarting curriculum stage {stage_label} (length <= {stage_suffix}) with {len(stage_train_df)} training samples"
    )

    stage_train_dataset = build_sft_dataset(stage_train_df)
    stage_val_dataset = build_sft_dataset(stage_val_df) if not stage_val_df.empty else None

    stage_examples = sample_examples(stage_val_df, stage_train_df)
    callbacks = []
    if stage_examples:
        callbacks.append(
            TranslationEvalCallback(
                tokenizer=tokenizer,
                samples=stage_examples,
                instruction_template=INSTRUCTION,
                stage_name=f"Stage {stage_label}",
            )
        )

    stage_training_args = build_stage_training_args(stage_name, stage_output_dir)
    stage_peft_config = peft_config if (peft_config is not None and final_trainer is None) else None

    trainer = SFTTrainer(
        model=current_model,
        args=stage_training_args,
        train_dataset=stage_train_dataset,
        eval_dataset=stage_val_dataset,
        peft_config=stage_peft_config,
        callbacks=callbacks if callbacks else None,
    )

    trainer.train()
    current_model = trainer.model
    final_trainer = trainer

if final_trainer is None:
    raise RuntimeError("No curriculum stage contained training data. Check the length thresholds and dataset.")

print("\nCurriculum training complete. Evaluating on the full validation set...")

full_val_dataset = build_sft_dataset(val_df)
eval_results = final_trainer.evaluate(eval_dataset=full_val_dataset)
val_loss = eval_results.get("eval_loss", float("nan"))

# Save final LoRA adapter
final_trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)


if lora:
    # Reload base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        config=new_config
    )

    # Load LoRA adapter on top
    lora_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)

    # Merge and unload adapters
    merged_model = lora_model.merge_and_unload()

    # Save merged full model
    MERGED_OUTPUT_DIR = OUTPUT_DIR + "-merged"
    merged_model.save_pretrained(MERGED_OUTPUT_DIR)
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)

    print(f"Merged full model saved to {MERGED_OUTPUT_DIR}")
else:
    MERGED_OUTPUT_DIR = OUTPUT_DIR

hf_repo_id = OUTPUT_DIR.split("/")[-1]
hf_repo_id = f"leobitz/{hf_repo_id}-merged"
hf_token = os.environ.get("HF_TOKEN")

api = HfApi(token=hf_token)
create_repo(repo_id=hf_repo_id, exist_ok=True, token=hf_token, private=True)
api.upload_folder(
    repo_id=hf_repo_id,
    folder_path=str(MERGED_OUTPUT_DIR),
    path_in_repo=".",
    commit_message=f"Performance {val_loss:.4f}",
)
print(f"Pushed model and tokenizer to https://huggingface.co/{hf_repo_id}")