import os
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from dataset import ImageCaptionDataset  

def collate_fn(batch_data):
    images, captions = zip(*batch_data)
    
    images = torch.stack(images)
    
    return images, captions

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_loaders(batch_size=32, valid=0.1, train=0.8, root_dir="./data/local", num_workers=0, pin_memory=False):
    file_path = "./datasets/image_caption_dataset.pth"
    
    dataset = ImageCaptionDataset(root_dir=root_dir)
    torch.save(dataset, file_path)
    
    train_size = int(train * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                             batch_size=batch_size,
                             sampler=train_sampler,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             collate_fn=collate_fn)
    
    valid_loader = DataLoader(trainset,
                             batch_size=batch_size,
                             sampler=valid_sampler,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             collate_fn=collate_fn)
    
    test_loader = DataLoader(testset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            collate_fn=collate_fn)
    
    return train_loader, valid_loader, test_loader
    
if __name__ == "__main__":
  dataloader,_ , _= get_loaders(batch_size=32, valid=0.0, train=1.0, root_dir="/data/jiachenghao/DDPM_inversion/data/local/", num_workers=0, pin_memory=False)
  
  i = 0
  for images, captions in dataloader:
      print(images.shape)  
      print(captions[0], len(captions))  
      
      if(i>10):
        break
      
      i+=1