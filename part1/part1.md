# LLM
## part 1
> ChatGLM-6B 是清华与智谱AI开源的LLM模型。官网地址：[https://chatglm.cn/blog](https://chatglm.cn/blog)
> 目前已ChatGLM-6B已开源。github：[https://github.com/THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)。

### try
```
# package
git clone https://github.com/THUDM/ChatGLM-6B.git
cd ChatGLM-6B
pip install -r requirements.txt
pip install rouge_chinese nltk jieba datasets

# code
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()  # 半精度
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
```
