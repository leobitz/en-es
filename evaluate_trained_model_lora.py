import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import tqdm
import evaluate
import torch
from transformers import AutoModelForCausalLM
from openai import OpenAI
from openai import OpenAI, AsyncOpenAI
import asyncio
import threading
import atexit
from transformers import set_seed
import os
import json
import argparse
import requests
import subprocess
import time
from dotenv import load_dotenv
load_dotenv()
set_seed(42)
token = os.getenv("HF_TOKEN")


parser = argparse.ArgumentParser(description="Evaluate translation model.")
parser.add_argument(
    "--model-name",
    default="HuggingFaceTB/SmolLM-135M",
    help="Hugging Face model identifier to evaluate.",
)
# take temprature as argument
parser.add_argument(
    "--temperature",
    type=float,
    default=0.2,
    help="Temperature for sampling during generation.",
)
# top_p argument
parser.add_argument(
    "--top_p",
    type=float,
    default=0.8,
    help="Top-p (nucleus) sampling parameter during generation.",
)

model_name = parser.parse_args().model_name
temperature = parser.parse_args().temperature
top_p = parser.parse_args().top_p
print(model_name)

num_layers = 0
num_task_layers = 0
num_frozen_layers = 0
# leobitz/lora-qlora-frozen-18-task-layers-0-aug-0.2
if "frozen" in model_name and "task-layers" in model_name:
    num_task_layers = int(model_name.split("task-layers-")[1].split("-")[0])
    num_frozen_layers = int(model_name.split("frozen-")[1].split("-task-layers-")[0])
    num_layers = num_task_layers + num_frozen_layers

df = pd.read_parquet("exp-data/en-es.parquet")
batch_size = 16

tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)


tokenizer_args = {
    "truncation": True,
    "padding": "longest",
}
device='cuda' if torch.cuda.is_available() else 'cpu'

def translate_batch(texts, model, tokenizer, max_text_length=None):
    if max_text_length is None:
        tok_args = {**tokenizer_args, 'max_length': tokenizer.model_max_length}
    else:
        tok_args = {**tokenizer_args, 'max_length': max_text_length}
    inputs = tokenizer(texts, return_tensors="pt", **tok_args)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(**inputs)
    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return translations


sacrebleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")

def get_eval_metrics(references, predictions):
    assert len(references) == len(predictions), f"Length mismatch: {len(references)} vs {len(predictions)}"
    references = [[ref] for ref in references]
    sacrebleu_score = sacrebleu.compute(predictions=predictions, references=references)["score"]
    chrf_score = chrf.compute(predictions=predictions, references=references)["score"]
    return {
        "sacrebleu": sacrebleu_score,
        "chrf": chrf_score,
    }

def evaluate(dataset: pd.DataFrame, text_col: str, ref_col: str, translator, save_col=None) -> pd.DataFrame:
    translations = []
    for i in tqdm.tqdm(range(0, len(dataset), batch_size)):
        batch_texts = dataset[text_col].iloc[i:i+batch_size].tolist()
        batch_translations = translator(batch_texts)
        translations.extend([t.strip() for t in batch_translations])

    dataset[save_col] = translations
    references = dataset[ref_col].tolist()
    
    result = {}
    datasets_names = dataset['dataset'].unique().tolist()
    for name in datasets_names:
        sub_dataset = dataset[dataset['dataset'] == name].reset_index(drop=True)
        sub_references = sub_dataset[ref_col].tolist()
        sub_predictions = sub_dataset[save_col].tolist()
        result[name] = get_eval_metrics(sub_references, sub_predictions)
    result["all"] = get_eval_metrics(references, translations)
    return result



datasets_names=['corpus-en-es', 'scientific_papers_en_es', 'Document-Translation-en-es', 'medical-translation-test-set']
test_df = df[df['dataset'].isin(datasets_names)].reset_index(drop=True)
max_per_dataset = 500
all_test_df = test_df[test_df['split'] == 'test'].reset_index(drop=True)
all_test_df['tokenizer-len'] = all_test_df['EN'].apply(lambda x: len(tokenizer(x)['input_ids']))
# # length less than 1024 tokens
all_test_df = all_test_df[all_test_df['tokenizer-len'] < 512].reset_index(drop=True)
test_dfs = []
for name in datasets_names:
    sub_df = all_test_df[all_test_df['dataset'] == name].reset_index(drop=True)
    print(f"{name}: {len(sub_df)} samples")
    if len(sub_df) > max_per_dataset:
        sub_df = sub_df.sample(n=max_per_dataset, random_state=42).reset_index(drop=True)
    test_dfs.append(sub_df)
test_df = pd.concat(test_dfs).reset_index(drop=True)

def plain_prompting(text):
    return f"""
    Translate the following English text to Spanish.".
    English: {text}
    Spanish:
    """

def few_shot_prompting(text):
    prompt = (
        "Translate the following English text to Spanish.\n\n"
        "English: Hello, how are you?\n"
        "Spanish: Hola, ¿cómo estás? [END]\n\n"
        "English: What is your name?\n"
        "Spanish: ¿Cómo te llamas? [END]\n\n"
        f"English: {text}\n"
        "Spanish:"
    )
    return prompt

def euro_llm_prompting(text):
    return f"English: {text} Spanish: "

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoConfig
)
from transformers import LlamaForCausalLM

print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
new_config = AutoConfig.from_pretrained(model_name, token=token)
if num_frozen_layers > 0:
    new_config.num_hidden_layers = num_task_layers + num_frozen_layers
model = LlamaForCausalLM.from_pretrained(model_name, config=new_config, token=token).to(device)
print(len(model.model.layers))
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
print("Model loaded.")
def proc_prompt(prompt):
    return model.generate(prompt, temperature=temperature, top_p=top_p, max_new_tokens=512, stop="<|END|>")[0]['generated_text']

def proc_batch(texts, prompt_fn=euro_llm_prompting):
    prompts = [prompt_fn(t) for t in texts]
    return [proc_prompt(p) for p in prompts]
print("Starting evaluation...")
result = evaluate(
    test_df,
    text_col='EN',
    ref_col='ES',
    translator=proc_batch,
    save_col='translation',
)

eval_dir_name = "eval-results"
os.makedirs(eval_dir_name, exist_ok=True)

with open(os.path.join(eval_dir_name, f"{model_name.replace('/', '_')}_evaluation.json"), "w") as f:
    json.dump(result, f, indent=4)
