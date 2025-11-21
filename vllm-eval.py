# serve_with_my_model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from vllm import LLM
from vllm.entrypoints.openai.api_server import run_server
from vllm.engine.arg_utils import EngineArgs
from llama import LlamaForCausalLM
from llama import MultiTaskLlamaForCausalLM
from llama_config import MyLlamaConfig
# 1. Load and modify your model exactly how you want
model_name = "leobitz/multi-task-True-frozen-6-task-layers-2"   # or local path

def register_models():
    # Register config (optional but clean)
    AutoConfig.register("multitask_llama", MyLlamaConfig)

    # Register the actual model class — this is the important one!
    AutoModelForCausalLM.register(
        config_class=MyLlamaConfig,   # or your config
        model_class=MultiTaskLlamaForCausalLM
    )

# Auto-register when the package is imported
register_models()

# 2. Wrap it with vLLM's LLM class (this moves it to GPU)
print("Handing model to vLLM...")
llm = LLM(
    model=model_name,           # ← your already-loaded model
    # tokenizer=tokenizer,   # ← your tokenizer
    max_model_len=2048,
    gpu_memory_utilization=0.90,
    tensor_parallel_size=1,          # change if multi-GPU
    enforce_eager=True,              # often required for custom models
    disable_log_stats=False,
    trust_remote_code=True,
)

# 3. Launch the exact same OpenAI-compatible server
print("Starting OpenAI API server on port 8000...")
run_server(
    model=llm,             # ← pass the vLLM LLM object directly!
    host="0.0.0.0",
    port=8000,
    # all other args are now inherited from the LLM object
)