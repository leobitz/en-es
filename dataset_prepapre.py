from datasets import load_dataset
import pandas as pd
import random
import os

dfs = []
max_per_dataset = 50000


ds = load_dataset("miracFence/scientific_papers_en_es")
df = ds["train"].to_pandas()
# rename text_no_abstract to EN
df = df.rename(columns={"text_no_abstract": "EN", "translated": "ES"})
df['dataset'] = 'scientific_papers_en_es'
df = df[['dataset', 'EN', 'ES']]
# remove null on ES or EN
print("before", df.shape)
df = df.dropna(subset=['EN', 'ES'])
# apply strip to EN and ES
df['EN'] = df['EN'].str.strip()
df['ES'] = df['ES'].str.strip()
# remove empty strings
df = df[(df['EN'] != '') & (df['ES'] != '')]
df['split'] = 'test'
print("after", df.shape)
df = df[['dataset', 'split', 'EN', 'ES']].reset_index(drop=True)
dfs.append(df)


ds = load_dataset("kudo-research/mustc-en-es-text-only")
train_df = ds["train"].to_pandas()
train_df = train_df.sample(n=min(len(train_df), max_per_dataset), random_state=42)
val_df = ds["dev"].to_pandas()
test_df = ds["test"].to_pandas()
train_df['split'] = 'train'
val_df['split'] = 'validation'
test_df['split'] = 'test'
df = pd.concat([train_df, val_df, test_df], ignore_index=True)
df['dataset'] = 'mustc-en-es-text-only'
df['EN'] = df['translation'].apply(lambda x: x['en']).str.strip()
df['ES'] = df['translation'].apply(lambda x: x['es']).str.strip()
df = df[['dataset', 'EN', 'ES', 'split']]
# remove null on ES or EN
print("before", df.shape)
df = df.dropna(subset=['EN', 'ES'])
# remove empty strings
df = df[(df['EN'] != '') & (df['ES'] != '')]
print("after", df.shape)
df = df[['dataset', 'split', 'EN', 'ES']].reset_index(drop=True)
dfs.append(df)



ds = load_dataset("loresiensis/corpus-en-es")
train_df = ds["train"].to_pandas()
train_df = train_df.sample(n=min(len(train_df), max_per_dataset), random_state=42)
test_df = ds["test"].to_pandas()
train_df['split'] = 'train'
test_df['split'] = 'test'
df = pd.concat([train_df, test_df], ignore_index=True)
df['dataset'] = 'corpus-en-es'
df = df[['dataset', 'EN', 'ES', 'split']]
# remove null on ES or EN
print("before", df.shape)
df = df.dropna(subset=['EN', 'ES'])
# apply strip to EN and ES
df['EN'] = df['EN'].str.strip()
df['ES'] = df['ES'].str.strip()
# remove empty strings
df = df[(df['EN'] != '') & (df['ES'] != '')]
print("after", df.shape)
df = df[['dataset', 'split', 'EN', 'ES']].reset_index(drop=True)
dfs.append(df)



ds = load_dataset("Thermostatic/OPUS-books-EN-ES")
df = ds["train"].to_pandas()
df = df.sample(n=min(len(df), max_per_dataset), random_state=42)
# English to EN and Spanish to ES
df = df.rename(columns={"English": "EN", "Spanish": "ES"})
df['dataset'] = 'OPUS-books-EN-ES'
df = df[['dataset', 'EN', 'ES']]
# remove null on ES or EN
print("before", df.shape)
df = df.dropna(subset=['EN', 'ES'])
# apply strip to EN and ES
df['EN'] = df['EN'].str.strip()
df['ES'] = df['ES'].str.strip()
# remove empty strings
df = df[(df['EN'] != '') & (df['ES'] != '')]
df['split'] = ['train' if random.random() < 0.8 else 'test' for _ in range(len(df))]
print("after", df.shape)
df = df[['dataset', 'split', 'EN', 'ES']].reset_index(drop=True)
dfs.append(df)




ds = load_dataset("ignacioct/wikipedia_en_es_m2m")
train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()
train_df['split'] = 'train'
test_df['split'] = 'test'
df = pd.concat([train_df, test_df], ignore_index=True)
df['dataset'] = 'wikipedia_en_es_m2m'
# en_sentence to EN and es_sentence to ES
df = df.rename(columns={"en_sentence": "EN", "es_sentence": "ES"})
df = df[['dataset', 'EN', 'ES', 'split']]
# remove null on ES or EN
print("before", df.shape)
df = df.dropna(subset=['EN', 'ES'])
# apply strip to EN and ES
df['EN'] = df['EN'].str.strip()
df['ES'] = df['ES'].str.strip()
# remove empty strings
df = df[(df['EN'] != '') & (df['ES'] != '')]
print("after", df.shape)
df = df[['dataset', 'split', 'EN', 'ES']].reset_index(drop=True)
dfs.append(df)



ds = load_dataset("Iker/Document-Translation-en-es")
df = ds["train"].to_pandas()
df['dataset'] = 'Document-Translation-en-es'
# es to ES and en to EN
df = df.rename(columns={"en": "EN", "es": "ES"})
df = df[['dataset', 'EN', 'ES']]
# remove null on ES or EN
print("before", df.shape)
df = df.dropna(subset=['EN', 'ES'])
# apply strip to EN and ES
df['EN'] = df['EN'].str.strip()
df['ES'] = df['ES'].str.strip()
# remove empty strings
df = df[(df['EN'] != '') & (df['ES'] != '')]
# split into train and test by 80% train and 20% test

df['split'] = ['train' if random.random() < 0.8 else 'test' for _ in range(len(df))]
print("after", df.shape)
df = df[['dataset', 'split', 'EN', 'ES']].reset_index(drop=True)
dfs.append(df)


ds = load_dataset("aimped/medical-translation-test-set")
df = ds["en_es"].to_pandas()
df['dataset'] = 'medical-translation-test-set'
# source to EN and target to ES
df = df.rename(columns={"source": "EN", "target": "ES"})
df = df[['dataset', 'EN', 'ES']]
# remove null on ES or EN
print("before", df.shape)
df = df.dropna(subset=['EN', 'ES'])
# apply strip to EN and ES
df['EN'] = df['EN'].str.strip()
df['ES'] = df['ES'].str.strip()
# remove empty strings
print("after", df.shape)
df = df[(df['EN'] != '') & (df['ES'] != '')]
df['split'] = 'test'
print("after", df.shape)
df = df[['dataset', 'split', 'EN', 'ES']].reset_index(drop=True)
dfs.append(df)




ds = load_dataset("kirchik47/english-spanish-translator")
df = ds["train"].to_pandas()
df = df.sample(n=min(len(df), max_per_dataset), random_state=42)
df['dataset'] = 'english-spanish-translator'
# sentences_en to EN and sentences_es to ES
df = df.rename(columns={"sentences_en": "EN", "sentences_es": "ES"})
df = df[['dataset', 'EN', 'ES']]
# remove null on ES or EN
print("before", df.shape)
df = df.dropna(subset=['EN', 'ES'])
# apply strip to EN and ES
df['EN'] = df['EN'].str.strip()
df['ES'] = df['ES'].str.strip()
# remove empty strings
df = df[(df['EN'] != '') & (df['ES'] != '')]
df['split'] = ['train' if random.random() < 0.8 else 'test' for _ in range(len(df))]
print("after", df.shape)
df = df[['dataset', 'split', 'EN', 'ES']].reset_index(drop=True)
dfs.append(df)


df = pd.concat(dfs, ignore_index=True)
# remove null
df = df.dropna(subset=['EN', 'ES'])
# remove duplicates
# apply strip to EN and ES
df['EN'] = df['EN'].str.strip()
df['ES'] = df['ES'].str.strip()
# remove empty strings
df = df[(df['EN'] != '') & (df['ES'] != '')]
df = df.drop_duplicates(subset=['EN', 'ES']).reset_index(drop=True)

df['split'].value_counts()


# save to exp-data/en-es.parquet

os.makedirs("exp-data", exist_ok=True)
df.to_parquet("exp-data/en-es.parquet", index=False)