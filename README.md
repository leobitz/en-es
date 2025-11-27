# en-es

End-to-end pipeline for preparing bilingual English→Spanish data, fine-tuning SmolLM/SmolLM2 variants (multi-task, LoRA, QLoRA, curriculum), and benchmarking checkpoints with vLLM-backed evaluation suites.

## Getting Started
- **Python**: 3.10+ with CUDA-enabled PyTorch build for training/eval on GPUs.
- **Dependencies**: create a virtualenv/conda env, then `pip install -r requirements.txt`.
- **Access tokens**: run `huggingface-cli login` and export `HF_TOKEN` if you plan to push checkpoints (`export HF_TOKEN=hf_...`).
- **Folders**: scripts expect `exp-data/` to exist; it is created automatically when running the data prep utilities.

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Repository Tour
- `dataset_prepapre.py` – CLI script that downloads multiple public EN↔ES corpora, cleans them, and writes `exp-data/en-es.parquet`.
- `prepare-dataset.ipynb` / `augment-dataset.ipynb` – Jupyter notebooks for exploratory cleaning and for generating synthetic pairs saved under `exp-data/synthetic-en-es-data.parquet`.
- `train_hf.py` – baseline Hugging Face Trainer recipe over the curated dataset; supports freezing layers, adding fresh task-specific blocks, and toggling LoRA/QLoRA heads.
- `train_hf_augmented.py` – identical to `train_hf.py` but mixes in synthetic data controlled by `--aug-size`.
- `train_hf_augmented_curriculum.py` – augmented recipe plus dynamic max-length curriculum (see `length_schedule`) to ease optimization on long sentences.
- `sft_train.py` / `sft_train_curriculum.py` – TRL `SFTTrainer` alternatives that operate on SmolLM2 with optional LoRA, bf16, and curriculum gating; best when you want TRL logging.
- `train.sh`, `train_aug.sh`, `train_curr.sh` – convenience sweep launchers that call the Python scripts with the grid used in the paper/logs.
- `evaluate_trained_model_vllm.py` – spins up a vLLM server for a given Hugging Face model, streams evaluation prompts, and stores SacreBLEU/chrF metrics under `eval-results/`.
- `llm-eval.sh` – sample batch of `evaluate_trained_model_vllm.py` invocations at different temperatures/top-p settings.
- `process-results.ipynb` – notebook for aggregating JSON metrics and plotting curves.

## Preparing Data
1. **Aggregate public corpora**
   ```bash
   python dataset_prepapre.py
   ```
   This pulls the listed Hugging Face datasets, normalizes column names to `['dataset', 'split', 'EN', 'ES']`, and writes `exp-data/en-es.parquet`.
2. **Create train/val parquet** – the training scripts lazily derive `exp-data/en-es-train-val.parquet` the first time they run; no extra step needed unless you want to inspect it.
3. **Optional synthetic augmentation** – use `augment-dataset.ipynb` (or any custom generator) to create `exp-data/synthetic-en-es-data.parquet`. The augmented training scripts sample from it according to `--aug-size`.

## Training Recipes
All training commands assume the environment above plus CUDA-capable hardware.

### Hugging Face Trainer baselines (`train_hf.py`)
```bash
python train_hf.py \
  --epochs 5 \
  --num-frozen-layers 24 \
  --num-task-layers 2 \
  --use-lora
```
- `--num-frozen-layers` controls how many original transformer blocks stay frozen.
- `--num-task-layers` adds fresh randomly initialized layers on top (trainable by default).
- `--use-lora` / `--use-qlora` inject PEFT adapters instead of new blocks.
- Outputs live in `exp-data/runs/<run_name>` and (if `HF_TOKEN` is set) the script uploads the best checkpoint to `leobitz/<run_name>`.

### Augmented data variant (`train_hf_augmented.py`)
```bash
python train_hf_augmented.py \
  --epochs 8 \
  --num-frozen-layers 18 \
  --num-task-layers 0 \
  --use-lora \
  --aug-size 0.5
```
Adds up to `0.5 * len(train_proc_df)` synthetic rows sampled from `exp-data/synthetic-en-es-data.parquet`, deduplicates pairs, and retrains with the blended corpus.

### Curriculum + augmentation (`train_hf_augmented_curriculum.py`)
```bash
python train_hf_augmented_curriculum.py \
  --epochs 12 \
  --num-frozen-layers 14 \
  --num-task-layers 2 \
  --use-lora \
  --aug-size 1.0
```
Identical flags to `train_hf_augmented.py`, but it gradually ramps the allowed sequence length following `length_schedule` (short sentences early, full 512 tokens later) to stabilize long-context learning.

### TRL SFT pipelines (`sft_train.py` and `sft_train_curriculum.py`)
```bash
python sft_train.py \
  --num_frozen_layers 28 \
  --num_task_layers 2 \
  --lora \
  --bf16 \
  --aug_size 1.0 \
  --epochs 10
```
- These scripts wrap TRL’s `SFTTrainer`, giving finer control over bf16, LoRA target modules, and synthetic sampling.
- Curriculum counterpart mirrors the above but follows a staged length schedule similar to the HF Trainer curriculum script.

### Batch launchers
When sweeping several configurations, run the helper shell scripts instead of copy/pasting commands:
```bash
bash train.sh          # LoRA baselines
bash train_aug.sh      # LoRA + synthetic mixes
bash train_curr.sh     # Curriculum variants
```
Each file is a simple list of the Python invocations shown above; edit them to match your grid search.

## Evaluating with vLLM
`evaluate_trained_model_vllm.py` automates spinning up a vLLM server, streaming prompts, and shutting everything down cleanly.

1. **Ensure GPU memory** – the default command boots vLLM with `--max-model-len 2048` and `--gpu-memory-utilization 0.95`. Adjust if your GPU is smaller.
2. **Login for datasets** – the script loads `google/wmt24pp` plus several HF corpora; make sure `huggingface-cli login` has access.
3. **Run evaluation**
   ```bash
   python evaluate_trained_model_vllm.py \
     --model-name leobitz/LoRA-F24-T0-aug1.0-bf161-merged \
     --temperature 0.3 \
     --top_p 0.9
   ```
   - The script starts `vllm serve` for the requested `--model-name`, waits until `http://localhost:8000/v1/models` responds, and then streams batches through the `euro_llm_prompting` template.
   - Generated translations plus SacreBLEU/chrF scores are saved to `eval-results/<model>_T{temperature}_P{top_p}_evaluation.json`.
   - vLLM logs land in `exp-data/logs/vllm_stdout.log` and `exp-data/logs/vllm_stderr.log` for later inspection.
4. **Batch runs** – edit and execute `bash llm-eval.sh` to reproduce the temperature/top-p grid used in prior experiments.

The evaluation script automatically tears down the vLLM subprocess even if an exception occurs. If you prefer to keep a long-running server, start it manually and comment out the spawn/stop section before invoking `proc_batch`.

## Results Exploration
- All evaluation JSON files live under `eval-results/` (see the precomputed ones already in the repo).
- Use `process-results.ipynb` to compare runs visually.
- Raw training artifacts (checkpoints, tokenizer snapshots, Trainer logs) are staged under `exp-data/runs/` and `exp-data/hf-artifacts/`.

Happy translating! Let us know if additional training regimes or evaluation datasets would be helpful.# en-es
