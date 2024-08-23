import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path).convert('RGB'))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = image.permute(2, 0, 1)
    
    return image


class ImageCaptionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Generate caption using BLIP
        '''
        inputs = self.processor(images=image, return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        '''
        
        offsets=(0,0,0,0)
        image = load_512(np.array(image)[:, :, :3], *offsets)
        
        
        txt_name = img_name.replace('.jpg', '.txt')
        txt_path = os.path.join(self.root_dir, txt_name)
        
        #self.save_caption(txt_path, caption)
        
        with open(txt_path, 'r') as f:
            caption = f.read().strip()  
            #print(caption, txt_path) 
        
        return image, caption
        
    def save_caption(self, txt_path, caption):
        with open(txt_path, 'w') as f:
            f.write(caption)

