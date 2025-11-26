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

train_dataset = Dataset.from_pandas(train_df)
val_dataset   = Dataset.from_pandas(val_df)

# Instruction template for translation
INSTRUCTION = "English: {en} Spanish:"

def formatting_prompts_func(example):
    text = INSTRUCTION.format(en=example["EN"]) + " " + example["ES"] + tokenizer.eos_token
    return {"text": text}

train_dataset = train_dataset.map(formatting_prompts_func)
val_dataset   = val_dataset.map(formatting_prompts_func)

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,  # Will be handled by DataCollatorForLanguageModeling
    )

example_samples = val_dataset.select(range(3))

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
val_dataset   = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)


class TranslationEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, val_dataset, num_samples=3):
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset.select(range(min(num_samples * 10, len(val_dataset))))  # small pool
        self.num_samples = num_samples

    def generate_translation(self, model, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,

            )
        full_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Extract only the Spanish part after "Spanish:"
        try:
            spanish = full_text.split("Spanish:")[1].strip().split(tokenizer.eos_token)[0].strip()
        except:
            spanish = full_text.split("Spanish:")[1].strip() if "Spanish:" in full_text else "ERROR"
        return spanish

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()

        print("\n" + "="*80)
        print(f"END OF EPOCH {state.epoch:.1f} - SAMPLE TRANSLATIONS")
        print("="*80)


        for i, sample in enumerate(example_samples, 1):
            prompt = INSTRUCTION.format(en=sample["EN"])

            # base_pred = self.generate_translation(self.base_model, prompt)
            current_pred = self.generate_translation(model, prompt)
            reference = sample["ES"]

            # print(f"\nSample {i}:")
            print(f"EN → {sample['EN']}")
            print(f"REF → {reference}")
            # print(f"BASE (SmolLM2-135M) → {base_pred}")
            print(f"CURRENT (Fine-tuned) → {current_pred}")
            print("-"*80)

        model.train()

# ==========================
# Training arguments
# ==========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    logging_steps=1000,
    save_strategy="epoch",
    eval_strategy="epoch",
    fp16=False,
    bf16=True,
    report_to=["wandb"],
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    remove_unused_columns=False,
    run_name=RUN_NAME,
    weight_decay=WEIGHT_DECAY,
    save_total_limit=0,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir=f"exp-data/runs/{RUN_NAME}/logs",
    max_grad_norm=GRAD_CLIP_VAL,
    dataloader_num_workers=DATALOADER_WORKERS,
    logging_strategy="steps",
    disable_tqdm=False,  
)

# ==========================
# SFTTrainer
# ==========================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    callbacks=[TranslationEvalCallback(tokenizer, val_dataset, num_samples=3)],
)

# ==========================
# Start training
# ==========================
print("Starting fine-tuning for English → Spanish translation with LoRA...")
trainer.train()

# Save final LoRA adapter
trainer.save_model(OUTPUT_DIR)
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

val_loss = trainer.evaluate()['eval_loss']

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