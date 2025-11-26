
import argparse
import os
import copy
from typing import Dict, List, Optional, Tuple
import pandas as pd
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments
)
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
# from transformers import LlamaForCausalLM
from llama import LlamaForCausalLM
from peft import LoraConfig, get_peft_model
from transformers import set_seed
from huggingface_hub import HfApi, create_repo
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from transformers.trainer_callback import PrinterCallback
set_seed(42)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("WANDB_PROJECT", "en-es")

length_schedule: List[Tuple[int, int]] = [
    (0, 64),
    (3, 128),
    (4, 256),
    (5, 384),
    (6, 480),
    (7, 512),
]

def parse_args():
    parser = argparse.ArgumentParser(description="Train EN->ES translation model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--num-frozen-layers", type=int, default=8, help="Count of frozen base layers")
    parser.add_argument("--num-task-layers", type=int, default=2, help="Number of task-specific layers")
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA fine-tuning")
    parser.add_argument("--use-qlora", action="store_true", help="Enable QLoRA fine-tuning")
    parser.add_argument("--aug-size", type=float, default=0.1, help="percentage of augmented data to use compared to original data")
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
aug_size = args.aug_size

is_multi_task = (not use_lora)  and (not use_qlora)
method_name = "multi-task" if is_multi_task else "lora-qlora"
run_name = f"{method_name}-frozen-{num_frozen_layers}-task-layers-{num_task_layers}-aug-{aug_size}_curriculum"
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


def ensure_length_column(df: pd.DataFrame, tokenizer: AutoTokenizer) -> pd.DataFrame:
    if "length" in df.columns and df["length"].notna().all():
        return df

    def _compute_length(row: pd.Series) -> int:
        sample_text = template.format(english_text=row["EN"], spanish_text=row["ES"])
        return len(tokenizer.encode(sample_text, add_special_tokens=False))

    df = df.copy()
    df["length"] = df.apply(_compute_length, axis=1)
    return df

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
    synthetic_df = pd.read_parquet("exp-data/synthetic-en-es-data.parquet")
    original_synthetic_size = len(synthetic_df)
    size = int(len(train_proc_df) * aug_size)
    size = min(size, len(synthetic_df))
    synthetic_df = synthetic_df.sample(n=size, random_state=42).reset_index(drop=True)
    print(f"Using {len(synthetic_df)} augmented samples out of {original_synthetic_size} available.")
    train_proc_df = pd.concat([train_proc_df, synthetic_df]).reset_index(drop=True)
    
    train_df = train_proc_df[train_proc_df["split"] == "train"]
    train_df = train_df.sample(min(max_train_sample_size, len(train_df)), random_state=42).reset_index(drop=True)
    val_df = train_proc_df[train_proc_df["split"] == "val"]
    val_df = val_df.sample(min(max_val_sample_size, len(val_df)), random_state=42).reset_index(drop=True)

    # remove duplicates in train_df based on EN and ES columns
    train_df = train_df.drop_duplicates(subset=["EN", "ES"]).reset_index(drop=True)
    val_df = val_df.drop_duplicates(subset=["EN", "ES"]).reset_index(drop=True)

    print(f"Number of samples: {len(train_df)}")
    print(f"Number of validation samples: {len(val_df)}")


train_df = ensure_length_column(train_df, tokenizer)
val_df = ensure_length_column(val_df, tokenizer)


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        initial_length_limit: Optional[int] = None,
        enable_curriculum: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.absolute_max_length = max_length
        self.enable_curriculum = enable_curriculum
        self.template = "English: {} Spanish: "
        self.current_length_limit = min(initial_length_limit or max_length, max_length)
        self.active_indices: List[int] = []
        self._recompute_active_indices(force=True)

    def _recompute_active_indices(self, force: bool = False) -> bool:
        if not self.enable_curriculum:
            new_indices = list(range(len(self.df)))
        else:
            mask = self.df["length"] <= self.current_length_limit
            new_indices = self.df.index[mask].tolist()
            if not new_indices:
                raise ValueError(
                    f"No samples available for max length {self.current_length_limit}. Check length_schedule."
                )

        changed = force or new_indices != getattr(self, "active_indices", [])
        if changed:
            self.active_indices = new_indices
        return changed

    def set_length_limit(self, new_limit: int) -> bool:
        capped_limit = min(new_limit, self.absolute_max_length)
        if not self.enable_curriculum or capped_limit == self.current_length_limit:
            self.current_length_limit = capped_limit
            return False
        self.current_length_limit = capped_limit
        return self._recompute_active_indices()

    def __len__(self) -> int:
        return len(self.active_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row_idx = self.active_indices[idx]
        row = self.df.iloc[row_idx]
        en_text = row["EN"].strip()
        es_text = row["ES"].strip() + " <|END|>"
        prompt = self.template.format(en_text)

        max_tokens = min(self.current_length_limit, self.absolute_max_length)
        prompt_ids = self.tokenizer(
            prompt,
            truncation=True,
            max_length=max_tokens,
            add_special_tokens=False,
        )["input_ids"]

        remaining_tokens = max_tokens - len(prompt_ids)
        if remaining_tokens > 0:
            es_ids = self.tokenizer(
                es_text,
                truncation=True,
                max_length=remaining_tokens,
                add_special_tokens=False,
            )["input_ids"]
        else:
            es_ids = []

        input_ids = prompt_ids + es_ids
        labels = [-100] * len(prompt_ids) + es_ids

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def dynamic_padding_collator(tokenizer: AutoTokenizer):
    pad_token_id = tokenizer.pad_token_id

    def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not batch:
            return {
                "input_ids": torch.empty(0, dtype=torch.long),
                "attention_mask": torch.empty(0, dtype=torch.long),
                "labels": torch.empty(0, dtype=torch.long),
            }

        max_len = max(item["input_ids"].size(0) for item in batch)
        batch_size = len(batch)
        input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for idx, item in enumerate(batch):
            seq_len = item["input_ids"].size(0)
            input_ids[idx, :seq_len] = item["input_ids"]
            attention_mask[idx, :seq_len] = 1
            label_len = item["labels"].size(0)
            labels[idx, :label_len] = item["labels"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return _collate


def get_max_length_for_epoch(epoch: int, schedule: List[Tuple[int, int]]) -> int:
    target = schedule[0][1]
    for start_epoch, max_len in schedule:
        if epoch >= start_epoch:
            target = max_len
        else:
            break
    return target


class CurriculumSchedulerCallback(TrainerCallback):
    def __init__(self, dataset: TranslationDataset, schedule: List[Tuple[int, int]]):
        self.dataset = dataset
        self.schedule = sorted(schedule, key=lambda item: item[0])
        self.current_length = None

    def _apply_schedule(self, epoch: int, control) -> None:
        target_length = get_max_length_for_epoch(epoch, self.schedule)
        if target_length == self.current_length:
            return
        length_changed = self.dataset.set_length_limit(target_length)
        if not length_changed:
            self.current_length = target_length
            return
        self.current_length = target_length
        if control is not None:
            control.should_recompute_train_dataloader = True
        print(f"[Curriculum] Epoch {epoch}: max length set to {target_length}")

    def on_train_begin(self, args, state, control, **kwargs):
        self._apply_schedule(epoch=0, control=control)

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch_number = int(state.epoch) if state.epoch is not None else 0
        self._apply_schedule(epoch=epoch_number, control=control)

initial_curriculum_length = get_max_length_for_epoch(0, length_schedule)
train_dataset = TranslationDataset(
    train_df,
    tokenizer,
    max_length=max_length,
    initial_length_limit=initial_curriculum_length,
    enable_curriculum=True,
)
val_dataset = TranslationDataset(
    val_df,
    tokenizer,
    max_length=max_length,
    initial_length_limit=max_length,
    enable_curriculum=False,
)
data_collator = dynamic_padding_collator(tokenizer)


sample_idx = 0
sample_row = train_df.iloc[train_dataset.active_indices[sample_idx]]
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
    logging_steps=1000,
    report_to=["wandb"],
    max_grad_norm=grad_clip_val,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    dataloader_num_workers=dataloader_workers,
    fp16=torch.cuda.is_available(),
    optim="adamw_torch",
    logging_strategy="steps",
    disable_tqdm=False,  
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

curriculum_callback = CurriculumSchedulerCallback(train_dataset, length_schedule)
trainer.add_callback(curriculum_callback)

trainer.add_callback(
    EarlyStoppingCallback(
        early_stopping_patience=2,
        early_stopping_threshold=0.001,
    )
)
trainer.remove_callback(PrinterCallback)

trainer.train()
best_model_path = trainer.state.best_model_checkpoint
print("Best model saved at:", best_model_path)

best_checkpoint = trainer.state.best_model_checkpoint
if best_checkpoint is not None:
    print(f"Loading best model from {best_checkpoint}")
    trainer._load_best_model()
else:
    print("No best checkpoint found; evaluating current model weights.")
eval_results = trainer.evaluate(eval_dataset=val_dataset)
val_loss = eval_results.get("eval_loss")
print(f"Validation loss: {val_loss}")

    
hf_repo_id = run_name
hf_repo_id = f"leobitz/{hf_repo_id}"
hf_token = os.environ.get("HF_TOKEN")

artifact_dir = Path("exp-data/hf-artifacts") / Path(run_name).name
artifact_dir.mkdir(parents=True, exist_ok=True)

model.save_pretrained(artifact_dir)
tokenizer.save_pretrained(artifact_dir)
model.config.save_pretrained(artifact_dir)

api = HfApi(token=hf_token)
create_repo(repo_id=hf_repo_id, exist_ok=True, token=hf_token, private=True)
api.upload_folder(
    repo_id=hf_repo_id,
    folder_path=str(artifact_dir),
    path_in_repo=".",
    commit_message=f"Upload {run_name} Performance {val_loss:.4f}",
)
print(f"Pushed model and tokenizer to https://huggingface.co/{hf_repo_id}")

