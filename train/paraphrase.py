import torch
from transformers import BertTokenizer, BertForMaskedLM

class SynonymGenerator:
    def __init__(self):
        # 加载预训练模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    
    def generate_synonyms(self, sentence):
        # 分词和编码
        tokens = self.tokenizer(sentence, return_tensors='pt')
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        
        # 随机选择一个词进行替换
        num_tokens = input_ids.size(1)
        masked_index = torch.randint(1, num_tokens-1, (1,)).item()  # 避免掩盖[CLS]和[SEP]标记
        input_ids[0, masked_index] = self.tokenizer.convert_tokens_to_ids('[MASK]')
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            predictions = outputs.logits[0, masked_index]
        
        # 获取预测的词
        predicted_ids = torch.topk(predictions, 5).indices
        predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_ids.tolist())
        
        # 替换掩盖词生成同义句
        synonyms = []
        for token in predicted_tokens:
            new_sentence_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
            new_sentence_tokens[masked_index] = token
            new_sentence = self.tokenizer.convert_tokens_to_string(new_sentence_tokens)
            synonyms.append(new_sentence)
        
        return synonyms

# 示例使用
generator = SynonymGenerator()
sentence = "The weather is nice today."
synonyms = generator.generate_synonyms(sentence)

for i, syn in enumerate(synonyms):
    print(f"Synonym {i+1}: {syn}")
