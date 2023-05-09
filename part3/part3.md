# LLM
## Part 3
```
对 ChatGLM-6B 模型基于 LoRA 进行finetune
以alpaca 为例
硬件需求
  显卡: 显存 >= 16G (最好24G或者以上)
  环境：
    python>=3.8
    cuda>=11.6, cupti, cuDNN, TensorRT等深度学习环境
```
### Ftuning with alpaca + LORA
DATA

[alpaca](https://github.com/tatsu-lab/stanford_alpaca): 提示（+输入）+ 输出
```
[
    {
        "instruction": "Give three tips for staying healthy.",
        "input": "",
        "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
    },...
]
```
PREPARE
```
# 转化alpaca数据集为jsonl
python cover_alpaca2jsonl.py --data_path data/alpaca_data.json --save_path data/alpaca_data.jsonl

# 生成数据 data/alpaca_data.jsonl
{"text": "### Instruction:\nIdentify the odd one out.\n\n### Input:\nTwitter, Instagram, Telegram\n\n### Response:\nTelegram\nEND\n"}
# 包含 Instruction、Input、Response 三个信息
# {"text": "### Instruction:\n【Instruction内容】\n\n### Input:\n【Input内容】\n\n### Response:\n【Response内容】\nEND\n"} 

# tokenize dataset 
python tokenize_dataset_rows.py  --jsonl_path data/alpaca_data.jsonl --save_path data/alpaca --max_seq_length 128
# 其中：input_ids = prompt_ids + target_ids + [config.eos_token_id]
# 返回：{"input_ids": input_ids, "seq_len": len(prompt_ids)}
```
TRAIN
```
python finetune.py \
    --dataset_path data/alpaca \
    --lora_rank 8 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --max_steps 52000 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir output

# {'train_runtime': 69720.3311, 'train_samples_per_second': 4.475, 'train_steps_per_second': 0.746, 'train_loss': 1.5100030215336726, 'epoch': 6.0}
```
PREDICT
```
python infer.py
```
