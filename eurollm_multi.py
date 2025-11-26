# train_en_to_es_sft.py

import torch
from datasets import load_dataset
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

from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import random
import numpy as np
import multiprocessing as mp
# ==========================
# Custom callback to print 3 samples every epoch
# ==========================
from transformers import TrainerCallback
from peft import PeftModel
# Set seed for reproducibility
set_seed(42)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("WANDB_PROJECT", "en-es")

num_frozen_layers = 28
num_task_layer = 2

RUN_NAME = f"smollm2-135m-en-es-multi-task-F{num_frozen_layers}-T{num_task_layer}"
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
DATASET_PATH = "exp-data/en-es-train-val.parquet"
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 10
OUTPUT_DIR = "./smollm2-135m-en-es-multi-task"
WEIGHT_DECAY = 0.01
GRAD_CLIP_VAL = 1.0
DATALOADER_WORKERS = mp.cpu_count() - 1 if mp.cpu_count() > 1 else 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
config.num_hidden_layers  = num_frozen_layers + num_task_layer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    config=config,
)

# Freeze all layers except the last `num_task_layer` in model.model.layers
for name, param in model.named_parameters():
    param.requires_grad = False

# Unfreeze the last `num_task_layer` transformer blocks
for layer in model.model.layers[-num_task_layer:]:
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

dataset = load_dataset("parquet", data_files=DATASET_PATH)["train"]
# sample 50%
# dataset = dataset.train_test_split(test_size=0.99, seed=42)["train"]

train_dataset = dataset.filter(lambda x: x["split"] == "train")
val_dataset   = dataset.filter(lambda x: x["split"] == "val")

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
        config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
        config.num_hidden_layers  = num_frozen_layers + num_task_layer
        self.base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            config=config,
        )
        self.base_model.eval()

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

            print(f"\nSample {i}:")
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
    save_total_limit=1,
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

val_loss = trainer.evaluate()['eval_loss']

hf_repo_id = OUTPUT_DIR.split("/")[-1]
hf_repo_id = f"leobitz/{hf_repo_id}-merged"
hf_token = os.environ.get("HF_TOKEN")

api = HfApi(token=hf_token)
create_repo(repo_id=hf_repo_id, exist_ok=True, token=hf_token, private=True)
api.upload_folder(
    repo_id=hf_repo_id,
    folder_path=OUTPUT_DIR,
    path_in_repo=".",
    commit_message=f"Performance {val_loss:.4f}",
)
print(f"Pushed model and tokenizer to https://huggingface.co/{hf_repo_id}")