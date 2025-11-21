
import argparse
import os
import copy
import pandas as pd
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    AutoConfig
)
from transformers.trainer_callback import EarlyStoppingCallback

from transformers import LlamaForCausalLM
from peft import LoraConfig, get_peft_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("WANDB_PROJECT", "en-es")

def parse_args():
    parser = argparse.ArgumentParser(description="Train EN->ES translation model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--num-frozen-layers", type=int, default=8, help="Count of frozen base layers")
    parser.add_argument("--num-task-layers", type=int, default=2, help="Number of task-specific layers")
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA fine-tuning")
    parser.add_argument("--use-qlora", action="store_true", help="Enable QLoRA fine-tuning")
    return parser.parse_args()

# exmaple
# python train_hf.py --epochs 5 --num-frozen-layers 10 --num-task-layers 4 --use-lora

args = parse_args()

num_frozen_layers = args.num_frozen_layers
num_task_layers = args.num_task_layers

max_length = 512
effective_batch_size = 32
batch_size = 32
accumulate_grad_batches = effective_batch_size // batch_size
weight_decay = 0.01
epochs = args.epochs
learning_rate = 5e-5
grad_clip_val = 1.0
max_train_sample_size = 100_000
max_val_sample_size = 10_000
model_name_or_path = "HuggingFaceTB/SmolLM-135M"
use_lora = args.use_lora
use_qlora = args.use_qlora
is_multi_task = (not use_lora)  and (not use_qlora)
method_name = "multi-task" if is_multi_task else "lora-qlora"
run_name = f"{method_name}-frozen-{num_frozen_layers}-task-layers-{num_task_layers}"
print(run_name)
os.environ.setdefault("WANDB_NAME", run_name)


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
config = AutoConfig.from_pretrained(model_name_or_path)
assert num_task_layers + num_frozen_layers <= config.num_hidden_layers, f"num_task_layers + num_frozen_layers must equal or less than {config.num_hidden_layers}"

if not use_lora and not use_qlora:
    pretrained_model = LlamaForCausalLM.from_pretrained(model_name_or_path)
    
    new_config = AutoConfig.from_pretrained(model_name_or_path)
    new_config.num_hidden_layers = num_task_layers + num_frozen_layers
    model = LlamaForCausalLM(new_config)

    random_task_layers = [copy.deepcopy(layer) for layer in model.model.layers[-num_task_layers:]]

    model.load_state_dict(pretrained_model.state_dict(), strict=False)

    # Overwrite the last num_task_layers layers with the random ones
    for i in range(num_task_layers):
        model.model.layers[-num_task_layers + i] = random_task_layers[i]
    
    for param in model.parameters():
        param.requires_grad = False

    if num_task_layers > 0:
        # Make the last num_task_layers layers trainable
        for layer in model.model.layers[-num_task_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

elif use_lora or use_qlora:
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    if use_qlora:
        # For QLoRA, quantize the model weights to 4-bit
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        pretrained_model = LlamaForCausalLM.from_pretrained(model_name_or_path, quantization_config=quantization_config)

        new_config = AutoConfig.from_pretrained(model_name_or_path)
        new_config.num_hidden_layers = num_task_layers + num_frozen_layers
        model = LlamaForCausalLM(new_config)
        
        model.load_state_dict(pretrained_model.state_dict(), strict=False)
    else:
        pretrained_model = LlamaForCausalLM.from_pretrained(model_name_or_path)
        new_config = AutoConfig.from_pretrained(model_name_or_path)
        new_config.num_hidden_layers = num_task_layers + num_frozen_layers
        model = LlamaForCausalLM(new_config)
        model.load_state_dict(pretrained_model.state_dict(), strict=False)

    model = get_peft_model(model, lora_config)
    print("Model updated with LoRA/QLoRA for parameter-efficient fine-tuning.")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

total_params = count_parameters(pretrained_model)
print(f"Total number of parameters in pretrained model: {total_params:,}")
total_params = count_parameters(model)
print(f"Total number of parameters in multi-task model: {total_params:,}")

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_non_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)

trainable_params = count_trainable_parameters(model)
non_trainable_params = count_non_trainable_parameters(model)
print(f"Trainable parameters: {trainable_params:,}")
print(f"Non-trainable parameters: {non_trainable_params:,}")

template = "English: {english_text} Spanish: {spanish_text} <|END|>"

# exp-data/en-es-train-val.parquet
if not os.path.exists("exp-data/en-es-train-val.parquet"):
    data_df = pd.read_parquet("exp-data/en-es.parquet")
    train_df = data_df[data_df["split"] == "train"].sample(frac=1.0, random_state=42).reset_index(drop=True)
    train_df['length'] = train_df.apply(lambda x: len(tokenizer.encode(template.format(english_text=x['EN'], spanish_text=x['ES']))), axis=1)
    train_df = train_df[train_df['length'] <= max_length].reset_index(drop=True)
    print(f"Number of training samples after length filtering: {len(train_df)}")
    # split train_df 80-20 into train and validation
    split_idx = int(0.8 * len(train_df))
    train_df_split = train_df.iloc[:split_idx].reset_index(drop=True)
    val_df_split = train_df.iloc[split_idx:].reset_index(drop=True)
    train_df = train_df_split
    val_df = val_df_split
    train_df = train_df.sample(min(max_train_sample_size, len(train_df)), random_state=42).reset_index(drop=True)
    val_df = val_df.sample(min(max_val_sample_size, len(val_df)), random_state=42).reset_index(drop=True)
    val_df['split'] = 'val'
    train_df['split'] = 'train'
    print(f"Number of samples: {len(train_df)}")
    print(f"Number of validation samples: {len(val_df)}")
    # to parquet
    train_proc_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    train_proc_df.to_parquet("exp-data/en-es-train-val.parquet")
else:
    train_proc_df = pd.read_parquet("exp-data/en-es-train-val.parquet")
    train_df = train_proc_df[train_proc_df["split"] == "train"].reset_index(drop=True)
    val_df = train_proc_df[train_proc_df["split"] == "val"].reset_index(drop=True)
    print(f"Number of samples: {len(train_df)}")
    print(f"Number of validation samples: {len(val_df)}")


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = "English: {} Spanish: "

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        en_text = self.df.iloc[idx]["EN"].strip()
        es_text = self.df.iloc[idx]["ES"].strip()
        prompt = self.template.format(en_text)
        # full_text = template.format(english_text=en_text, spanish_text=es_text)
        
        es_text = es_text + " <|END|>"
        # Tokenize prompt and full text
        prompt_ids = self.tokenizer(prompt, truncation=True, max_length=self.max_length, add_special_tokens=False)["input_ids"]
        es_ids = self.tokenizer(es_text, truncation=True, max_length=self.max_length, add_special_tokens=False)["input_ids"]

        # Labels: -100 for prompt, actual tokens for output
        labels = [-100] * len(prompt_ids) + es_ids
        input_ids = prompt_ids + es_ids

        # Pad if necessary
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
        else:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

train_dataset = TranslationDataset(train_df, tokenizer, max_length=max_length)
val_dataset = TranslationDataset(val_df, tokenizer, max_length=max_length)


sample_idx = 0
sample_row = train_df.iloc[sample_idx]
actual_text = template.format(english_text=sample_row['EN'], spanish_text=sample_row['ES'])
print("Actual text:", actual_text)

sample = train_dataset[sample_idx]
trimmed_input_ids = [tok_id for tok_id in sample["input_ids"].tolist() if tok_id != tokenizer.pad_token_id]
decoded_text = tokenizer.decode(trimmed_input_ids, skip_special_tokens=False)
print("Decoded text:", decoded_text)

# label
trimmed_label_ids = [tok_id for tok_id in sample["labels"].tolist() if tok_id != -100]
decoded_label_text = tokenizer.decode(trimmed_label_ids, skip_special_tokens=False)
print("Decoded label text:", decoded_label_text)

print("Match:", actual_text.strip() == decoded_text.strip())
print("Label Match:", sample_row['ES'].strip() == decoded_label_text.replace("<|END|>", "").strip())


cpu_count = os.cpu_count() or 1
dataloader_workers = max(cpu_count - 1, 1)

training_args = TrainingArguments(
    output_dir=f"exp-data/runs/{run_name}",
    run_name=run_name,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=accumulate_grad_batches,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir=f"exp-data/runs/{run_name}/logs",
    logging_steps=10,
    report_to=["wandb"],
    max_grad_norm=grad_clip_val,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    dataloader_num_workers=dataloader_workers,
    fp16=torch.cuda.is_available(),
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

trainer.add_callback(
    EarlyStoppingCallback(
        early_stopping_patience=2,
        early_stopping_threshold=0.001,
    )
)

trainer.train()


