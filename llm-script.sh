temperature=0.7
top_p=0.85

# python evaluate_trained_model_vllm.py --model-name utter-project/EuroLLM-1.7B --temperature $temperature --top_p $top_p

# python evaluate_trained_model_vllm.py --model-name leobitz/multi-task-frozen-6-task-layers-2-aug-1.0 --temperature $temperature --top_p $top_p

# python evaluate_trained_model_vllm.py --model-name leobitz/lora-qlora-frozen-18-task-layers-0-aug-1.0-merged --temperature $temperature --top_p $top_p
# python evaluate_trained_model_vllm.py --model-name leobitz/lora-qlora-frozen-18-task-layers-0-aug-0.2-merged --temperature $temperature --top_p $top_p

# python evaluate_trained_model_vllm.py --model-name leobitz/multi-task-frozen-16-task-layers-2-aug-1.0 --temperature $temperature --top_p $top_p
# python evaluate_trained_model_vllm.py --model-name leobitz/multi-task-frozen-16-task-layers-2-aug-0.2 --temperature $temperature --top_p $top_p

# python evaluate_trained_model_vllm.py --model-name leobitz/lora-qlora-frozen-6-task-layers-0-merged --temperature $temperature --top_p $top_p
# python evaluate_trained_model_vllm.py --model-name leobitz/lora-qlora-frozen-10-task-layers-0-merged --temperature $temperature --top_p $top_p
# python evaluate_trained_model_vllm.py --model-name leobitz/lora-qlora-frozen-16-task-layers-0-merged --temperature $temperature --top_p $top_p
# python evaluate_trained_model_vllm.py --model-name leobitz/lora-qlora-frozen-22-task-layers-0-merged --temperature $temperature --top_p $top_p
# python evaluate_trained_model_vllm.py --model-name leobitz/lora-qlora-frozen-28-task-layers-0-merged --temperature $temperature --top_p $top_p

# python evaluate_trained_model.py --model-name HuggingFaceTB/SmolLM-135M --temperature $temperature --top_p $top_p
# python evaluate_trained_model_vllm.py --model-name leobitz/multi-task-frozen-6-task-layers-2 --temperature $temperature --top_p $top_p
# python evaluate_trained_model.py --model-name leobitz/multi-task-frozen-10-task-layers-2 --temperature $temperature --top_p $top_p
# python evaluate_trained_model.py --model-name leobitz/multi-task-frozen-16-task-layers-2 --temperature $temperature --top_p $top_p
# python evaluate_trained_model.py --model-name leobitz/multi-task-frozen-22-task-layers-2 --temperature $temperature --top_p $top_p
# python evaluate_trained_model.py --model-name leobitz/multi-task-frozen-28-task-layers-2 --temperature $temperature --top_p $top_p

vllm serve leobitz/smollm2-135m-en-es-lora-merged --host 0.0.0.0 --port 8000 --max-model-len 2048 --gpu-memory-utilization 0.95