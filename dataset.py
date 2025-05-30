# dataset.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pandas as pd
from transformers import T5Tokenizer

class SlakeDataset(Dataset):
    def __init__(self, tokenizer, image_root, annotation_file, max_length=64):
        self.tokenizer = tokenizer
        self.image_root = image_root
        self.max_length = max_length
        
        
        self.data = pd.read_json(annotation_file)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
  
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data.iloc[idx]
        image_path = os.path.join(self.image_root, entry['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        question = entry['question']
        answer = entry['answer']
        
       
        question_enc = self.tokenizer(question, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
       
        answer_enc = self.tokenizer(answer, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")

        return {
            'image': image,
            'input_ids': question_enc.input_ids.squeeze(0),
            'attention_mask': question_enc.attention_mask.squeeze(0),
            'labels': answer_enc.input_ids.squeeze(0),
            'image_filename': entry['image']
        }

class VQARADDataset(Dataset):
    def __init__(self, tokenizer, image_root, annotation_file, max_length=64):
        self.tokenizer = tokenizer
        self.image_root = image_root
        self.max_length = max_length
      
        df = pd.read_excel(annotation_file)
       
        self.data = df
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
          
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data.iloc[idx]
        image_path = os.path.join(self.image_root, entry['image_filename'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        question = entry['question']
        answer = entry['answer']

        question_enc = self.tokenizer(question, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        answer_enc = self.tokenizer(answer, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")

        return {
            'image': image,
            'input_ids': question_enc.input_ids.squeeze(0),
            'attention_mask': question_enc.attention_mask.squeeze(0),
            'labels': answer_enc.input_ids.squeeze(0),
            'image_filename': entry['image_filename']
        }
