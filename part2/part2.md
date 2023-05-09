# LLM
## Part 2
### Ptuning V2 with ADGEN
DATA

[AdvertiseGen](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1): 根据输入（content）生成一段广告词（summary）
```
# put it at ChatGLM-6B/ptuning/AdvertiseGen
{
"content": "类型#上衣*版型#宽松*版型#显瘦*图案#线条*衣样式#衬衫*衣袖型#泡泡袖*衣款式#抽绳",
"summary": "这件衬衫的款式非常的宽松，利落的线条可以很好的隐藏身材上的小缺点，穿在身上有着很好的显瘦效果。领口装饰了一个可爱的抽绳，漂亮的绳结展现出了十足的个性，配合时尚的泡泡袖型，尽显女性甜美可爱的气息。"
}
```
TRAIN
```
cd ptuning
bash train.sh

#  epoch                    =       0.42
#  train_loss               =     3.9492
#  train_runtime            = 3:19:40.95
#  train_samples            =     114599
#  train_samples_per_second =      4.006
#  train_steps_per_second   =       0.25
```
EVAL
```
bash evaluate.sh

#   predict_bleu-4             =     7.9679
#   predict_rouge-1            =      31.21
#   predict_rouge-2            =     7.1448
#   predict_rouge-l            =    25.0058
#   predict_runtime            = 0:46:11.11
#   predict_samples            =       1070
#   predict_samples_per_second =      0.386
#   predict_steps_per_second   =      0.386

评测指标为中文 Rouge score 和 BLEU-4。生成的结果保存在： ./output/adgen-chatglm-6b-pt-128-2e-2/generated_predictions.txt
Ptuning的模型在： ./output/adgen-chatglm-6b-pt-128-2e-2/checkpoint-3000
```
PREDICT
```
CHECKPOINT_PATH = "ChatGLM-6B/ptuning/output/adgen-chatglm-6b-pt-128-2e-2/checkpoint-3000"

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
# 加载新 Checkpoint（只包含 PrefixEncoder 参数）
config = AutoConfig.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, pre_seq_len=128)
model_ad = AutoModel.from_pretrained("THUDM/chatglm-6b", config=config, trust_remote_code=True).half().cuda()

prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model_ad.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
model_ad = model_ad.eval()

# compare
response, _ = model.chat(tokenizer, "类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞", history=[])
print(f"ori: {response}")
response, _ = model_ad.chat(tokenizer, "类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞", history=[])
print(f"ptuning: {response}") 
```
