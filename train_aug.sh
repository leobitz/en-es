# python sft_train.py --num_frozen_layers 28 --num_task_layers 2 --aug_size 1.0 --epochs 10
# python sft_train.py --num_frozen_layers 22 --num_task_layers 2 --aug_size 1.0 --epochs 10
# python sft_train.py --num_frozen_layers 14 --num_task_layers 2 --aug_size 1.0 --epochs 10
# python sft_train.py --num_frozen_layers 6 --num_task_layers 2 --aug_size 1.0 --epochs 10

python sft_train.py --num_frozen_layers 30 --num_task_layers 0 --aug_size 1.0 --epochs 10 --bf16 --lora
python sft_train.py --num_frozen_layers 24 --num_task_layers 0 --aug_size 1.0 --epochs 10 --bf16 --lora
python sft_train.py --num_frozen_layers 16 --num_task_layers 0 --aug_size 1.0 --epochs 10 --bf16 --lora
python sft_train.py --num_frozen_layers 8 --num_task_layers 0 --aug_size 1.0 --epochs 10 --bf16 --lora