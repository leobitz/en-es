temperature=0.0
top_p=1.0

# python evaluate_trained_model_vllm.py --model-name utter-project/EuroLLM-1.7B --temperature $temperature --top_p $top_p
# python evaluate_trained_model_vllm.py --model-name HuggingFaceTB/SmolLM-135M --temperature $temperature --top_p $top_p

# python evaluate_trained_model_vllm.py --model-name leobitz/MultiTask-F6-T2-aug1.0-bf160-merged --temperature $temperature --top_p $top_p
# python evaluate_trained_model_vllm.py --model-name leobitz/MultiTask-F14-T2-aug1.0-bf160-merged --temperature $temperature --top_p $top_p
# python evaluate_trained_model_vllm.py --model-name leobitz/MultiTask-F22-T2-aug1.0-bf160-merged --temperature $temperature --top_p $top_p
# python evaluate_trained_model_vllm.py --model-name leobitz/MultiTask-F28-T2-aug1.0-bf160-merged --temperature $temperature --top_p $top_p

# python evaluate_trained_model_vllm.py --model-name leobitz/LoRA-F8-T0-aug1.0-bf161-merged --temperature $temperature --top_p $top_p
# python evaluate_trained_model_vllm.py --model-name leobitz/LoRA-F16-T0-aug1.0-bf161-merged --temperature $temperature --top_p $top_p
# python evaluate_trained_model_vllm.py --model-name leobitz/LoRA-F24-T0-aug1.0-bf161-merged --temperature $temperature --top_p $top_p
# python evaluate_trained_model_vllm.py --model-name leobitz/LoRA-F30-T0-aug1.0-bf161-merged --temperature $temperature --top_p $top_p

# python evaluate_trained_model_vllm.py --model-name leobitz/MultiTask-F6-T2-aug0.0-bf160-merged --temperature $temperature --top_p $top_p 
# python evaluate_trained_model_vllm.py --model-name leobitz/MultiTask-F14-T2-aug0.0-bf160-merged --temperature $temperature --top_p $top_p 
python evaluate_trained_model_vllm.py --model-name leobitz/MultiTask-F22-T2-aug0.0-bf160-merged --temperature $temperature --top_p $top_p 
python evaluate_trained_model_vllm.py --model-name leobitz/MultiTask-F28-T2-aug0.0-bf160-merged --temperature $temperature --top_p $top_p 

python evaluate_trained_model_vllm.py --model-name leobitz/LoRA-F8-T0-aug0.0-bf161-merged --temperature $temperature --top_p $top_p 
python evaluate_trained_model_vllm.py --model-name leobitz/LoRA-F16-T0-aug0.0-bf161-merged --temperature $temperature --top_p $top_p
python evaluate_trained_model_vllm.py --model-name leobitz/LoRA-F24-T0-aug0.0-bf161-merged --temperature $temperature --top_p $top_p 
python evaluate_trained_model_vllm.py --model-name leobitz/LoRA-F30-T0-aug0.0-bf161-merged --temperature $temperature --top_p $top_p 

temperature=0.7
top_p=0.9

# python evaluate_trained_model_vllm.py --model-name utter-project/EuroLLM-1.7B --temperature $temperature --top_p $top_p
# python evaluate_trained_model_vllm.py --model-name HuggingFaceTB/SmolLM-135M --temperature $temperature --top_p $top_p
# python evaluate_trained_model_vllm.py --model-name leobitz/MultiTask-F6-T2-aug1.0-bf160-merged --temperature $temperature --top_p $top_p > exp-data/terminal/vllm_mt_f6_t2_aug1.0_bf160_temp0.7_topp0.9.txt
# python evaluate_trained_model_vllm.py --model-name leobitz/MultiTask-F14-T2-aug1.0-bf160-merged --temperature $temperature --top_p $top_p > exp-data/terminal/vllm_mt_f14_t2_aug1.0_bf160_temp0.7_topp0.9.txt
# python evaluate_trained_model_vllm.py --model-name leobitz/MultiTask-F22-T2-aug1.0-bf160-merged --temperature $temperature --top_p $top_p > exp-data/terminal/vllm_mt_f22_t2_aug1.0_bf160_temp0.7_topp0.9.txt
# python evaluate_trained_model_vllm.py --model-name leobitz/MultiTask-F28-T2-aug1.0-bf160-merged --temperature $temperature --top_p $top_p > exp-data/terminal/vllm_mt_f28_t2_aug1.0_bf160_temp0.7_topp0.9.txt

# python evaluate_trained_model_vllm.py --model-name leobitz/LoRA-F8-T0-aug1.0-bf161-merged --temperature $temperature --top_p $top_p > exp-data/terminal/vllm_lora_f8_t0_aug1.0_bf161_temp0.7_topp0.9.txt
# python evaluate_trained_model_vllm.py --model-name leobitz/LoRA-F16-T0-aug1.0-bf161-merged --temperature $temperature --top_p $top_p > exp-data/terminal/vllm_lora_f16_t0_aug1.0_bf161_temp0.7_topp0.9.txt
# python evaluate_trained_model_vllm.py --model-name leobitz/LoRA-F24-T0-aug1.0-bf161-merged --temperature $temperature --top_p $top_p > exp-data/terminal/vllm_lora_f24_t0_aug1.0_bf161_temp0.7_topp0.9.txt
# python evaluate_trained_model_vllm.py --model-name leobitz/LoRA-F30-T0-aug1.0-bf161-merged --temperature $temperature --top_p $top_p > exp-data/terminal/vllm_lora_f30_t0_aug1.0_bf161_temp0.7_topp0.9.txt
temperature=0.3
top_p=0.9

# python evaluate_trained_model_vllm.py --model-name utter-project/EuroLLM-1.7B --temperature $temperature --top_p $top_p
# python evaluate_trained_model_vllm.py --model-name HuggingFaceTB/SmolLM-135M --temperature $temperature --top_p $top_p


#  leobitz/LoRA-F24-T0-aug1.0-bf161-merged

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
