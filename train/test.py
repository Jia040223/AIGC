import os
import random
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.nn.parallel import DistributedDataParallel as DDP

class SynonymGenerator:
    def __init__(self, device):
        self.device = device
        # 加载预训练模型和分词器
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large').to(self.device)
        self.model = DDP(self.model, device_ids=[self.device])

    def generate_synonyms(self, sentences, num_return_sequences=5):
        # 构造生成任务的输入
        inputs = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(self.device)
        
        # 生成同义句
        outputs = self.model.module.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=256,
            num_return_sequences=num_return_sequences,
            num_beams=num_return_sequences,  # Use num_return_sequences for num_beams to ensure all variations are generated
            early_stopping=True
        )
        
        # 解码生成的句子
        decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        # 转换为每个句子5个同义句的列表
        return [decoded_outputs[i:i + num_return_sequences] for i in range(0, len(decoded_outputs), num_return_sequences)]

class TextDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with open(file_path, 'r', encoding='utf-8') as file:
            sentence = file.read().strip()
        return sentence

def replace_random_word(sentence):
    words = sentence.split()
    if len(words) > 1:
        word_to_replace = random.choice(words)
        replacement = f"{word_to_replace}_syn"  # Example replacement
        new_sentence = sentence.replace(word_to_replace, replacement, 1)
        return new_sentence
    return sentence

def process_files(input_folder, output_folder, generator, batch_size=8):
    # 获取所有.txt文件路径
    file_paths = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

    # 提取所有不带 '_' 的文件名，如 'x.txt'
    original_files = {f[:-4] for f in file_paths if not f[:-4].endswith('_')}

    # 提取所有带 '_' 的文件名，如 'x_.txt'
    underscore_files = {f[:-5] for f in file_paths if f[:-4].endswith('_')}
    print(len(original_files))
    print(len(underscore_files))

    # 找出有 x.txt 但没有 x_.txt 的文件
    file_paths = [os.path.join(input_folder, f + '.txt') for f in original_files if f not in underscore_files]
    print(len(file_paths))
    
    # 创建数据集和数据加载器
    dataset = TextDataset(file_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # 获取当前进程的rank
    rank = dist.get_rank()

    # 遍历数据加载器中的批次
    for batch_idx, batch in enumerate(dataloader):
        sentences = batch
        synonyms_batch = generator.generate_synonyms(sentences)
        
        for i, sentence in enumerate(sentences):
            output_path = os.path.join(output_folder, os.path.basename(file_paths[batch_idx * batch_size + i]).replace('.txt', '_.txt'))
            synonyms = synonyms_batch[i]
            #print(sentence)
            #print(synonyms, "\n")
            
            # 选择一个与原句不同的同义句
            synonym_to_save = None
            for syn in synonyms:
                if syn.strip() != sentence:
                    synonym_to_save = syn
                    break
            
            # 如果没有找到与原句不同的同义句，进行随机单词替换
            if synonym_to_save is None:
                synonym_to_save = replace_random_word(sentence)
            
            # 去除 `Paraphrase:` 标签
            if synonym_to_save.startswith("Paraphrase: "):
                synonym_to_save = synonym_to_save[len("Paraphrase: "):].strip()
            
            # 保存到新文件
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(synonym_to_save + '\n')
                print(sentence)
                print(synonym_to_save)

    print(f"Processing completed. Synonyms saved to '{output_folder}'.")

def main():
    # 初始化分布式环境
    dist.init_process_group(backend='nccl')
    
    # 获取当前进程的rank和总进程数
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    device = torch.device(f'cuda:{rank}')
    generator = SynonymGenerator(device=device)
    
    input_folder = './data/local'  # 输入文件夹路径
    output_folder = './data/local'  # 输出文件夹路径

    process_files(input_folder, output_folder, generator, batch_size=5)

if __name__ == "__main__":
    main()
