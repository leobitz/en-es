
import pandas as pd
from transformers import pipeline
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sacrebleu import corpus_bleu
import tqdm
import evaluate
import torch
from transformers import AutoModelForCausalLM


df = pd.read_parquet("exp-data/en-es.parquet")


df


batch_size = 16



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


df['dataset'].unique()


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


checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


datasets_names=['corpus-en-es', 'scientific_papers_en_es', 'Document-Translation-en-es', 'medical-translation-test-set']
test_df = df[df['dataset'].isin(datasets_names)].reset_index(drop=True)
max_per_dataset = 500
all_test_df = test_df[test_df['split'] == 'test'].reset_index(drop=True)
# all_test_df['tokenizer-len'] = all_test_df['EN'].apply(lambda x: len(tokenizer(x)['input_ids']))
# # length less than 1024 tokens
# all_test_df = all_test_df[all_test_df['tokenizer-len'] < 1024].reset_index(drop=True)
test_dfs = []
for name in datasets_names:
    sub_df = all_test_df[all_test_df['dataset'] == name].reset_index(drop=True)
    print(f"{name}: {len(sub_df)} samples")
    if len(sub_df) > max_per_dataset:
        sub_df = sub_df.sample(n=max_per_dataset, random_state=42).reset_index(drop=True)
    test_dfs.append(sub_df)
test_df = pd.concat(test_dfs).reset_index(drop=True)


test_df


# model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es", use_safetensors=True).eval().to(device)
# tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")


# def temp_pipe(texts):
#     return translate_batch(texts, model, tokenizer)

# result = evaluate(test_df, 
#                   text_col='EN', 
#                   ref_col='ES', 
#                   translator=temp_pipe, 
#                   save_col='translation')


# result


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



# checkpoint = "HuggingFaceTB/SmolLM-135M"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token_id = tokenizer.eos_token_id
# model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device).eval()


# def generate(model, tokenizer, device, prompts, system_prompt=None, max_new_tokens=50, return_full_text=False, **generate_kwargs):
#     if system_prompt == None:
#         system_prompt = "You are a helpful assistant that completes sentences."
#     prompts = [f"{system_prompt}\n\n{prompt}".strip() for prompt in prompts]
#     inputs = tokenizer(prompts, return_tensors="pt", padding="longest", truncation=True, max_length=512, padding_side='left').to(device)
#     temperature = generate_kwargs.pop("temperature", 0.7)
#     do_sample = temperature > 0.0
#     generate_kwargs.update({
#         "temperature": temperature,
#         "do_sample": do_sample,
#     })
#     output_ids = model.generate(**inputs, tokenizer=tokenizer, max_new_tokens=max_new_tokens, **generate_kwargs)
    
#     results = []
#     for i, prompt in enumerate(prompts):
#         prompt_length = inputs["input_ids"].shape[-1]
#         if not return_full_text:
#             generated_tokens = output_ids[i][prompt_length:]
#         else:
#             generated_tokens = output_ids[i]
#         results.append(tokenizer.decode(generated_tokens, skip_special_tokens=True).strip())
#     return results


# generate(model=model, 
#     tokenizer=tokenizer, 
#     device=device, 
#     prompts=[plain_prompting("my name is John.")],
#     system_prompt="You are a very accurate English to Spanish translator.", 
#     max_new_tokens=1024, 
#     temperature=0.0,
#     repetition_penalty=1.2,
#     stop_strings=["END"],
#     top_p=1.0)


# generate(model=model, 
#     tokenizer=tokenizer, 
#     device=device, 
#     prompts=[few_shot_prompting("my name is John.")],
#     system_prompt="You are a very accurate English to Spanish translator.", 
#     max_new_tokens=1024, 
#     temperature=0.0,
#     repetition_penalty=1.2,
#     stop_strings=["[END]"],
#     top_p=1.0)



from vllm import LLM, SamplingParams


if __name__ == "__main__":

    model_name = "HuggingFaceTB/SmolLM-135M"  # or any non-instruct HF model

    llm = LLM(model=model_name, max_model_len=1024)
    sampling_params = SamplingParams(max_tokens=102, temperature=0.0, top_p=1.0, repetition_penalty=1.2, stop=["[END]"])

    def generate_vllm(prompts):
        return llm.generate(prompts, sampling_params)

    def temp_pipe(texts):
        outs = generate_vllm(prompts=[few_shot_prompting(text) for text in texts])
        return [out.outputs[0].text for out in outs]

    result = evaluate(test_df, 
                    text_col='EN', 
                    ref_col='ES', 
                    translator=temp_pipe, 
                    save_col='translation')
