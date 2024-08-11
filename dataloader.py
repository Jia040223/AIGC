import torch
import torch.utils.data as Data
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import Bert_Dataset
import os


def collate_fn(batch_data, pad_token_id=0, pad_token_type_id=0, pad_label_id=0):
    input_ids_list, token_type_ids_list, attention_mask_list, label_list = [], [], [], []
    max_len = 0
    for input_ids, token_type_ids, labels in batch_data:
        input_ids_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        label_list.append(labels)

        max_len = max(max_len, len(input_ids))

    for i in range(len(input_ids_list)):
        cur_len = len(input_ids_list[i])
        input_ids_list[i] = input_ids_list[i] + [pad_token_id] * (max_len - cur_len)
        token_type_ids_list[i] = token_type_ids_list[i] + [pad_token_type_id] * (max_len - cur_len)
        label_list[i] = label_list[i] + [pad_label_id] * (max_len - cur_len)
        attention_mask_list.append([1] * cur_len + [0] * (max_len - cur_len))

    return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(label_list), torch.tensor(attention_mask_list)

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_loaders(label2id, batch_size=512, valid=0.0, train=0.8, filename="ChineseCorpus199801.txt",
                encoding="gbk", num_workers=0, pin_memory=False):
    file_path = "./datasets/bert_dataset.pth"
    if os.path.exists(file_path):
        dataset = torch.load(file_path)
    else:
        dataset = Bert_Dataset(filename=filename, lable2id=label2id, encoding=encoding)
        torch.save(dataset, file_path)
    
    train_size = int(train * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = Data.random_split(dataset, [train_size, test_size])

    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = Data.DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              collate_fn=collate_fn)
    
    valid_loader = Data.DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              collate_fn=collate_fn)
    
    test_loader = Data.DataLoader(testset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             collate_fn=collate_fn)
    
    return train_loader, valid_loader, test_loader
