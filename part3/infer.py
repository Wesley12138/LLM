# from modeling_chatglm import ChatGLMForConditionalGeneration
# import torch
# from peft import PeftModel
# from transformers import AutoTokenizer
# from cover_alpaca2jsonl import format_example

# torch.set_default_tensor_type(torch.cuda.HalfTensor)
# model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, device_map='auto')
# model = PeftModel.from_pretrained(model, "mymusise/chatGLM-6B-alpaca-lora")
# torch.set_default_tensor_type(torch.cuda.FloatTensor)
# tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import json
from cover_alpaca2jsonl import format_example

model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, load_in_8bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = PeftModel.from_pretrained(model, "./output/")
instructions = json.load(open("data/alpaca_data.json"))  # alpaca数据集

with torch.no_grad():
    for idx, item in enumerate(instructions[:3]):
        feature = format_example(item)
        input_text = feature['context']
        ids = tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids])
        out = model.generate(
            input_ids=input_ids,
            max_length=150,
            do_sample=False,
            temperature=0
        )
        out_text = tokenizer.decode(out[0])
        answer = out_text.replace(input_text, "").replace("\nEND", "").strip()
        item['infer_answer'] = answer
        print(out_text)
        print(f"### {idx+1}.Answer:\n", item.get('output'), '\n\n')
        # # answers.append({'index': idx, **item})